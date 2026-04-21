"""
Self-Pruning Neural Network — CUDA Implementation
Tredence Analytics AI Engineering Intern Case Study

Author  : <Your Name>
Dataset : CIFAR-10
Device  : CUDA (falls back to CPU if unavailable)

Architecture:
  - Custom PrunableLinear layer with learnable sigmoid gates
  - Sparsity regularisation via L1 penalty on gate values
  - Three-lambda sweep to show sparsity ↔ accuracy trade-off
  - Gate-value distribution plot saved as gate_distribution.png
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import GradScaler, autocast   # mixed-precision for speed

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Device Setup
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PrunableLinear – Custom Gated Linear Layer
# ──────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that learns which weights to zero-out.

    Each weight w_{ij} is multiplied by a learnable gate g_{ij} ∈ (0, 1):
        g_{ij}  = sigmoid( s_{ij} )
        y       = x @ (W ⊙ G)^T + b

    During training the sparsity loss drives most g → 0, effectively pruning
    the corresponding weights without any hard masking.

    Gradient flow:
        ∂L/∂W  = ∂L/∂y · G       (gates scale weight gradients)
        ∂L/∂s  = ∂L/∂y · W · σ'  (standard chain rule through sigmoid)
    Both paths are differentiable; autograd handles everything automatically.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # ── Primary weight & bias (same as nn.Linear) ──────────────────────
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # ── Gate scores: same shape as weight, also learned ────────────────
        # Initialise to 0 → sigmoid(0) = 0.5, so gates start at 50 % open.
        # The sparsity loss will then push most toward 0 during training.
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Kaiming uniform init for the weight (matches nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1 – compute gates ∈ (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)          # (out, in)

        # Step 2 – element-wise mask on the weight matrix
        pruned_weights = self.weight * gates             # (out, in)

        # Step 3 – standard affine transform
        return F.linear(x, pruned_weights, self.bias)

    def sparsity_info(self, threshold: float = 1e-2) -> dict:
        """Returns gate statistics (used for evaluation, not training)."""
        with torch.no_grad():
            gates = torch.sigmoid(self.gate_scores)
            total   = gates.numel()
            pruned  = (gates < threshold).sum().item()
        return {"total": total, "pruned": pruned,
                "sparsity": 100.0 * pruned / total}

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}")


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Self-Pruning Network
# ──────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32×3 → 10 classes).
    All linear layers use PrunableLinear so every weight has a gate.

    Architecture:
        Conv block  (shared feature extraction, NOT pruned)
        Flatten
        PrunableLinear 512 → 256  + ReLU + Dropout
        PrunableLinear 256 → 128  + ReLU + Dropout
        PrunableLinear 128 → 64   + ReLU
        PrunableLinear  64 → 10   (logits)
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        # ── Convolutional feature extractor ────────────────────────────────
        # Using standard Conv2d here so the pruning is demonstrated on the
        # dense fully-connected head (where most parameters usually live).
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2),                  nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                  nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),     # → (B, 256, 2, 2) = 1024-dim
        )

        flat_dim = 256 * 2 * 2   # 1024

        # ── Prunable fully-connected head ───────────────────────────────────
        self.classifier = nn.Sequential(
            PrunableLinear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            PrunableLinear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            PrunableLinear(256, 128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    # ── Helper: collect all PrunableLinear layers ──────────────────────────
    def prunable_layers(self) -> list:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values = Σ sigmoid(s_ij)  over every prunable layer.

        Why L1 encourages sparsity:
          The gradient of |g| w.r.t. g is ±1 (constant), regardless of
          magnitude.  Unlike L2 (gradient ∝ g), L1 keeps pushing small values
          all the way to zero rather than only asymptotically approaching it.
        """
        total = torch.tensor(0.0, device=DEVICE)
        for layer in self.prunable_layers():
            total = total + torch.sigmoid(layer.gate_scores).sum()
        return total

    def network_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall % of prunable weights whose gate < threshold."""
        total_w, pruned_w = 0, 0
        for layer in self.prunable_layers():
            info = layer.sparsity_info(threshold)
            total_w  += info["total"]
            pruned_w += info["pruned"]
        return 100.0 * pruned_w / total_w if total_w else 0.0

    def gate_values(self) -> np.ndarray:
        """Flatten all gate values to a 1-D numpy array (for plotting)."""
        vals = []
        with torch.no_grad():
            for layer in self.prunable_layers():
                vals.append(
                    torch.sigmoid(layer.gate_scores).cpu().numpy().ravel()
                )
        return np.concatenate(vals)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 256, num_workers: int = 4):
    """
    CIFAR-10 with standard augmentation for training and normalisation only
    for evaluation.  pin_memory=True enables faster CPU→GPU transfers.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_tf)

    pin = DEVICE.type == "cuda"
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, persistent_workers=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=True)

    return train_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: SelfPruningNet,
    loader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    lam: float,
) -> tuple[float, float, float]:
    """
    Returns (avg_total_loss, avg_cls_loss, avg_sparsity_loss).
    Uses AMP (Automatic Mixed Precision) for faster CUDA throughput.
    """
    model.train()
    total_loss_sum = cls_loss_sum = sp_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), \
                        labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with autocast(enabled=(DEVICE.type == "cuda")):
            logits    = model(images)
            cls_loss  = F.cross_entropy(logits, labels)
            sp_loss   = model.sparsity_loss()
            loss      = cls_loss + lam * sp_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Gradient clipping prevents exploding gradients (especially for gates)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_sum += loss.item()
        cls_loss_sum   += cls_loss.item()
        sp_loss_sum    += sp_loss.item()

    n = len(loader)
    return total_loss_sum / n, cls_loss_sum / n, sp_loss_sum / n


@torch.inference_mode()
def evaluate(model: SelfPruningNet, loader) -> float:
    """Returns top-1 accuracy (%) on the given dataloader."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE, non_blocking=True), \
                        labels.to(DEVICE, non_blocking=True)
        with autocast(enabled=(DEVICE.type == "cuda")):
            preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def train_model(
    lam: float,
    epochs: int = 40,
    lr: float = 3e-3,
    batch_size: int = 256,
) -> dict:
    """
    Full training run for a given λ.

    Returns a dict with:
        test_accuracy   – final top-1 % on CIFAR-10 test set
        sparsity        – % of prunable gates below 1e-2
        gate_values     – 1-D numpy array of all gate values
        history         – list of per-epoch dicts
    """
    print(f"\n{'='*60}")
    print(f"  Training  λ = {lam:.0e}   ({epochs} epochs)")
    print(f"{'='*60}")

    train_loader, test_loader = get_dataloaders(batch_size)

    model = SelfPruningNet(dropout=0.3).to(DEVICE)

    # Separate LR for gate_scores vs the rest (gates need gentle nudging)
    gate_params   = [p for n, p in model.named_parameters() if "gate_scores" in n]
    other_params  = [p for n, p in model.named_parameters() if "gate_scores" not in n]
    optimizer = optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": gate_params,  "lr": lr * 0.5},   # slower gate updates
    ], weight_decay=1e-4)

    # Cosine annealing with warm restarts keeps training from stagnating
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))

    history = []
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_l, cls_l, sp_l = train_one_epoch(
            model, train_loader, optimizer, scaler, lam
        )
        acc = evaluate(model, test_loader)
        sparsity = model.network_sparsity()
        scheduler.step()

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch, "total_loss": total_l,
            "cls_loss": cls_l, "sp_loss": sp_l,
            "accuracy": acc, "sparsity": sparsity,
        })

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Ep {epoch:3d}/{epochs}  "
                f"loss={total_l:.4f}  cls={cls_l:.4f}  sp={sp_l:.2f}  "
                f"acc={acc:.2f}%  sparse={sparsity:.1f}%  "
                f"({elapsed:.1f}s)"
            )

        if acc > best_acc:
            best_acc = acc

    final_acc     = evaluate(model, test_loader)
    final_sparsity = model.network_sparsity()

    print(f"\n  ▶  λ={lam:.0e}  |  Test Acc: {final_acc:.2f}%  "
        f"|  Sparsity: {final_sparsity:.1f}%")

    return {
        "lam":          lam,
        "test_accuracy": final_acc,
        "sparsity":      final_sparsity,
        "gate_values":   model.gate_values(),
        "history":       history,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(results: list[dict], save_path: str = "gate_distribution.png"):
    """
    For each λ, plot a histogram of the final gate values.
    A successful prune shows a sharp spike near 0 and a diffuse cluster near 1.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colours = ["#e74c3c", "#2ecc71", "#3498db"]

    for ax, res, colour in zip(axes, results, colours):
        gates = res["gate_values"]
        ax.hist(gates, bins=80, color=colour, alpha=0.82, edgecolor="white",
                linewidth=0.4)
        ax.set_title(
            f"λ = {res['lam']:.0e}\n"
            f"Acc: {res['test_accuracy']:.1f}%  |  "
            f"Sparse: {res['sparsity']:.1f}%",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Gate value  sigmoid(s)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(0.01, color="black", linestyle="--", linewidth=1.2,
                label="prune threshold (0.01)")
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle("Self-Pruning Network – Gate Value Distributions\n(CIFAR-10)",
                fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[INFO] Gate distribution plot saved → {save_path}")
    plt.close(fig)


def print_results_table(results: list[dict]):
    header = f"{'Lambda':<12} {'Test Accuracy':>16} {'Sparsity Level (%)':>20}"
    sep    = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in results:
        print(f"  {r['lam']:<10.0e}  {r['test_accuracy']:>14.2f}%  "
            f"{r['sparsity']:>18.1f}%")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main: Lambda Sweep
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Three λ values: low / medium / high  (as required by the problem)
    LAMBDAS = [1e-5, 1e-4, 1e-3]   # tweak if needed
    EPOCHS  = 40                   # increase to 60-80 for best results

    all_results = []
    for lam in LAMBDAS:
        result = train_model(lam=lam, epochs=EPOCHS)
        all_results.append(result)

    print_results_table(all_results)
    plot_gate_distribution(all_results, save_path="gate_distribution.png")

    # Find best model (highest accuracy)
    best = max(all_results, key=lambda r: r["test_accuracy"])
    print(f"\n[INFO] Best λ = {best['lam']:.0e}  "
        f"→  {best['test_accuracy']:.2f}% acc, {best['sparsity']:.1f}% sparse")


if __name__ == "__main__":
    main()