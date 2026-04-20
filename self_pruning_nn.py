""" Self-Pruning Neural Network on CIFAR-10 """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# PrunableLinear Layer

class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate parameters.

    Each weight w_ij has a corresponding gate_score g_ij.
    During the forward pass:
        gates       = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_w    = weight * gates                (element-wise)
        output      = input @ pruned_w.T + bias

    When a gate → 0, the corresponding weight is effectively removed.
    The optimizer updates both `weight` and `gate_scores` via backprop.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight & bias (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # Initialised at 0 so gates start at sigmoid(0)=0.5, closer to pruning boundary
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gates ∈ (0, 1); gradient flows through sigmoid into gate_scores
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise gating: effectively multiplies each weight by its gate
        pruned_weights = self.weight * gates

        # Standard linear operation — F.linear handles batches correctly
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from graph)."""
        return torch.sigmoid(self.gate_scores).detach()


# Network Definition

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32x32x3 → 10 classes).
    All linear projections use PrunableLinear.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3 * 32 * 32, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            PrunableLinear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            PrunableLinear(256, 128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # Flatten: (B, 3072)
        return self.net(x)

    def get_all_gates(self) -> torch.Tensor:
        """Concatenate gate tensors from every PrunableLinear layer."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.get_gates().cpu().flatten())
        return torch.cat(gates)

    def sparsity_loss(self) -> torch.Tensor:
        """
        Mean of all gate values (normalized L1 penalty).
        Dividing by total gate count keeps this in [0, 1],
        making it comparable to CrossEntropy and λ meaningful.
        Minimising this pulls gates toward 0, pruning weights.
        """
        total_sum = torch.tensor(0.0, device=next(self.parameters()).device)
        total_count = 0
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_sum = total_sum + gates.sum()
                total_count += gates.numel()
        return total_sum / total_count


# Data Loading

def get_cifar10_loaders(batch_size: int = 128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256,        shuffle=False, num_workers=2)
    return train_loader, test_loader


# Training & Evaluation

def train_one_epoch(model, loader, optimizer, device, lam: float):
    model.train()
    total_loss = correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        spar_loss = model.sparsity_loss()
        loss = cls_loss + lam * spar_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return correct / total


def compute_sparsity(model, threshold: float = 0.5) -> float:
    """Fraction of gates below `threshold` (considered pruned).
    Using 0.5 as threshold: gates below this are more 'off' than 'on'."""
    gates = model.get_all_gates()
    pruned = (gates < threshold).float().mean().item()
    return pruned


# Main Experiment

def run_experiment(lam: float, epochs: int, device, train_loader, test_loader):
    print(f"\n{'='*50}")
    print(f"  λ = {lam}   |   epochs = {epochs}")
    print(f"{'='*50}")

    model = SelfPruningNet().to(device)
    # Separate gate_scores from weights: weight_decay on gates fights the sparsity
    # gradient and causes mean_gate to freeze at 0.5 (sigmoid(0)). Exclude them.
    gate_params   = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    optimizer = torch.optim.Adam([
        {'params': weight_params, 'weight_decay': 1e-4},
        {'params': gate_params,   'weight_decay': 0.0},
    ], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, lam)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            sparsity  = compute_sparsity(model)
            mean_gate = model.get_all_gates().mean().item()
            print(f"  Epoch {epoch:3d} | loss {tr_loss:.4f} | "
                  f"train acc {tr_acc:.3f} | sparsity {sparsity:.2%} | mean_gate {mean_gate:.3f}")

    test_acc  = evaluate(model, test_loader, device)
    sparsity  = compute_sparsity(model)
    gates     = model.get_all_gates().numpy()

    print(f"\n Final test accuracy : {test_acc:.4f}")
    print(f" Sparsity level : {sparsity:.2%}")
    return test_acc, sparsity, gates


def plot_gate_distribution(gates_dict: dict, best_lam: float):
    best_gates = gates_dict[best_lam]

    fig, axes = plt.subplots(1, len(gates_dict), figsize=(5 * len(gates_dict), 4),
                             constrained_layout=True)
    if len(gates_dict) == 1:
        axes = [axes]

    for ax, (lam, gates) in zip(axes, gates_dict.items()):
        ax.hist(gates, bins=60, color='steelblue' if lam != best_lam else 'darkorange',
                edgecolor='white', linewidth=0.4)
        ax.set_title(f'λ = {lam}', fontsize=13)
        ax.set_xlabel('Gate value', fontsize=11)
        ax.set_ylabel('Count' if ax == axes[0] else '', fontsize=11)
        ax.axvline(0.01, color='red', linestyle='--', linewidth=1, label='threshold')
        ax.legend(fontsize=9)

    fig.suptitle('Distribution of Final Gate Values\n(orange = best model)',
                 fontsize=14, fontweight='bold')
    plt.savefig('gate_distributions.png', dpi=150, bbox_inches='tight')
    print("\n Saved gate_distributions.png")
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    lambdas = [0.1, 0.5, 1.0]      # stronger values to actively drive gates down
    epochs  = 30

    results = {}
    gates_dict = {}

    for lam in lambdas:
        test_acc, sparsity, gates = run_experiment(
            lam, epochs, device, train_loader, test_loader
        )
        results[lam] = (test_acc, sparsity)
        gates_dict[lam] = gates


# Summary Table
    print("\n\n" + "="*55)
    print(f"{'Lambda':<12} {'Test Accuracy':>14} {'Sparsity Level':>16}")
    print("-"*55)
    for lam, (acc, spar) in results.items():
        print(f"{lam:<12} {acc:>14.4f} {spar:>15.2%}")
    print("="*55)

    best_lam = max(results, key=lambda l: results[l][0])
    print(f"\n  Best model: λ = {best_lam} "
          f"(acc={results[best_lam][0]:.4f}, sparsity={results[best_lam][1]:.2%})")

# Plot
    plot_gate_distribution(gates_dict, best_lam)

if __name__ == '__main__':
    main()