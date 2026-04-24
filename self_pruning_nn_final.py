#!/usr/bin/env python3
"""
Self-Pruning Neural Network — CIFAR-10  (FINAL)
L0 Regularisation via Hard Concrete gates (Louizos, Welling & Kingma, ICLR 2018)

Extended from self_pruning_nn.py:
  * Per-epoch per-seed history logging
  * history.json saved after every lambda (crash safety)
  * plot_training_curves(history) — train_acc vs val_acc per epoch
  * plot_sparsity_curve(history)  — sparsity over epochs
"""

import os
import json
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Global config ─────────────────────────────────────────────────────────────
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE        = device
THRESHOLD     = 0.01
N_SEEDS       = 3
WARMUP_EPOCHS = 5
LAM_L2        = 1e-4
LAMBDAS       = [0.0, 0.1, 0.3, 1.0, 2.0, 5.0]
EPOCHS        = 30
NUM_EPOCHS    = EPOCHS


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Hard Concrete mixin
# ═════════════════════════════════════════════════════════════════════════════

class _HardConcreteMixin:
    BETA  = 2.0 / 3.0
    GAMMA = -0.1
    ZETA  =  1.1
    _LOG_RATIO = BETA * float(np.log(-GAMMA / ZETA))

    def _sample_gate(self, log_alpha: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps = 1e-6
            u = torch.rand_like(log_alpha).clamp(eps, 1.0 - eps)
            s = torch.sigmoid(
                (torch.log(u) - torch.log1p(-u) + log_alpha) / self.BETA
            )
        else:
            s = torch.sigmoid(log_alpha)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)

    def l0_penalty(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha - self._LOG_RATIO).sum()

    def l2_reg(self) -> torch.Tensor:
        p = torch.sigmoid(self.log_alpha - self._LOG_RATIO)
        return (p * self.weight.pow(2)).sum()

    @torch.no_grad()
    def gates(self) -> torch.Tensor:
        s = torch.sigmoid(self.log_alpha)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0.0, 1.0)

    @torch.no_grad()
    def sparsity_pct(self, threshold: float = THRESHOLD) -> float:
        return (self.gates() < threshold).float().mean().item() * 100.0

    @torch.no_grad()
    def exact_zero_pct(self) -> float:
        return (self.gates() == 0.0).float().mean().item() * 100.0


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PrunableLinear
# ═════════════════════════════════════════════════════════════════════════════

class PrunableLinear(_HardConcreteMixin, nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, init_log_alpha: float = 2.0):
        nn.Module.__init__(self)
        self.in_features  = in_features
        self.out_features = out_features

        self.weight    = nn.Parameter(torch.empty(out_features, in_features))
        self.log_alpha = nn.Parameter(torch.empty_like(self.weight))
        self.bias      = nn.Parameter(torch.zeros(out_features)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.normal_(self.log_alpha, mean=init_log_alpha, std=0.01)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self._sample_gate(self.log_alpha), self.bias)

    @property
    def gate_scores(self) -> nn.Parameter:
        return self.log_alpha

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PrunableConv2d
# ═════════════════════════════════════════════════════════════════════════════

class PrunableConv2d(_HardConcreteMixin, nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, init_log_alpha: float = 2.0):
        nn.Module.__init__(self)
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding

        self.weight    = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.log_alpha = nn.Parameter(torch.empty_like(self.weight))
        self.bias      = nn.Parameter(torch.zeros(out_channels)) if bias else None

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.normal_(self.log_alpha, mean=init_log_alpha, std=0.01)
        if self.bias is not None:
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1.0 / np.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight * self._sample_gate(self.log_alpha),
            self.bias,
            self.stride,
            self.padding,
        )

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"k={self.kernel_size}, s={self.stride}, p={self.padding}")


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Network Architecture
# ═════════════════════════════════════════════════════════════════════════════

class SelfPruningNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            PrunableConv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            PrunableConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            PrunableConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            PrunableLinear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            PrunableLinear(512, 256),
            nn.ReLU(inplace=True),
            PrunableLinear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

    def _prunable(self):
        return [m for m in self.modules()
                if isinstance(m, (PrunableLinear, PrunableConv2d))]

    @torch.no_grad()
    def all_gates(self) -> torch.Tensor:
        return torch.cat([m.gates().cpu().flatten() for m in self._prunable()])

    @torch.no_grad()
    def global_sparsity(self, threshold: float = THRESHOLD) -> float:
        g = self.all_gates()
        return (g < threshold).float().mean().item() * 100.0

    @torch.no_grad()
    def weight_stats(self, threshold: float = THRESHOLD) -> dict:
        prunable    = self._prunable()
        prun_total  = sum(m.weight.numel() for m in prunable)
        all_g       = self.all_gates()
        pruned_ct   = int((all_g < threshold).sum().item())
        prun_active = prun_total - pruned_ct
        exact_zero  = int((all_g == 0.0).sum().item())

        inference_total = sum(
            p.numel() for n, p in self.named_parameters()
            if not n.endswith("log_alpha")
        )
        non_prun = inference_total - prun_total

        effective     = prun_active + non_prun
        true_compress = inference_total / effective if effective > 0 else float("inf")
        head_compress = prun_total / prun_active   if prun_active > 0 else float("inf")

        return dict(
            prunable_total       = prun_total,
            prunable_active      = prun_active,
            pruned               = pruned_ct,
            exact_zero           = exact_zero,
            total_params         = inference_total,
            non_prunable         = non_prun,
            true_compression     = true_compress,
            headline_compression = head_compress,
            sparsity_pct         = 100.0 * pruned_ct / prun_total,
            exact_zero_pct       = 100.0 * exact_zero / prun_total,
        )


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Loss helpers
# ═════════════════════════════════════════════════════════════════════════════

def l0_loss(model: SelfPruningNet) -> torch.Tensor:
    return sum(m.l0_penalty() for m in model._prunable())


def l2_gate_scaled_loss(model: SelfPruningNet) -> torch.Tensor:
    return sum(m.l2_reg() for m in model._prunable())


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Data
# ═════════════════════════════════════════════════════════════════════════════

def build_dataloaders(batch_size: int = 128, root: str = "./data",
                      val_frac: float = 0.1, seed: int = 42):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root, train=True, download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root, train=False, download=True, transform=eval_tf
    )

    n_val   = int(len(full_train) * val_frac)
    n_train = len(full_train) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=gen)
    val_ds.dataset = copy.copy(full_train)
    val_ds.dataset.transform = eval_tf

    nw = 0 if os.name == "nt" else 2
    kw = dict(num_workers=nw, pin_memory=torch.cuda.is_available())
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, **kw)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw)
    return train_dl, val_dl, test_dl


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Training & Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion,
                    lam_effective: float, device):
    model.train()
    total_loss = correct = seen = 0
    n_train = len(loader.dataset)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits   = model(x)
        cls_loss = criterion(logits, y)

        l0   = l0_loss(model)
        l2   = l2_gate_scaled_loss(model)
        loss = cls_loss + (lam_effective / n_train) * l0 + LAM_L2 * l2

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        seen       += x.size(0)

    return total_loss / seen, 100.0 * correct / seen


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = seen = 0
    for x, y in loader:
        x, y   = x.to(device), y.to(device)
        logits  = model(x)
        loss    = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        seen       += x.size(0)
    return total_loss / seen, 100.0 * correct / seen


# ═════════════════════════════════════════════════════════════════════════════
# 8.  Inference benchmark
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def benchmark_inference(model, device, batch_sizes=(1, 128),
                        n_warmup: int = 20, n_reps: int = 200) -> dict:
    model.eval()
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 32, 32, device=device)
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_reps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        results[bs] = (float(np.mean(times)), float(np.std(times)))
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 9.  Single-seed experiment
# ═════════════════════════════════════════════════════════════════════════════

def _run_single_seed(lam: float, train_dl, val_dl, test_dl,
                     num_epochs: int, lr: float,
                     ckpt_path: str) -> dict:
    model     = SelfPruningNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    gate_params, other_params = [], []
    for name, p in model.named_parameters():
        (gate_params if name.endswith("log_alpha") else other_params).append(p)

    optimizer = optim.Adam(
        [
            {"params": other_params, "weight_decay": 1e-4},
            {"params": gate_params,  "weight_decay": 0.0},
        ],
        lr=lr,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # === HISTORY LOGGING ===
    log = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
        "sparsity":   [],
    }
    best_val_acc = -1.0

    for epoch in range(1, num_epochs + 1):
        ramp    = min(epoch / max(WARMUP_EPOCHS, 1), 1.0)
        lam_eff = lam * ramp

        tr_loss, tr_acc = train_one_epoch(
            model, train_dl, optimizer, criterion, lam_eff, DEVICE
        )
        va_loss, va_acc = evaluate(model, val_dl, criterion, DEVICE)
        scheduler.step()

        sp = model.global_sparsity()

        # === HISTORY LOGGING ===
        log["train_loss"].append(float(tr_loss))
        log["train_acc"].append(float(tr_acc))
        log["val_loss"].append(float(va_loss))
        log["val_acc"].append(float(va_acc))
        log["sparsity"].append(float(sp))

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    _, final_test_acc = evaluate(model, test_dl, criterion, DEVICE)
    stats = model.weight_stats()
    bench = benchmark_inference(model, DEVICE)

    return dict(
        model        = model,
        test_acc     = final_test_acc,
        best_val_acc = best_val_acc,
        stats        = stats,
        bench        = bench,
        log          = log,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 10.  Multi-seed experiment runner
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(lam: float, train_dl, val_dl, test_dl,
                   num_epochs: int = NUM_EPOCHS, lr: float = 1e-3,
                   n_seeds: int = N_SEEDS) -> dict:
    seeds = [42, 0, 1, 7, 13][:n_seeds]
    label = f"lam={lam}"
    print(f"\n{'-' * 64}\n  Experiment: {label}  ({n_seeds} seeds)\n{'-' * 64}")

    os.makedirs("results/ckpts", exist_ok=True)
    seed_results = []

    # === HISTORY LOGGING ===
    seed_history = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        ckpt = f"results/ckpts/best_{label}_seed{seed}.pt"
        r    = _run_single_seed(lam, train_dl, val_dl, test_dl,
                                num_epochs, lr, ckpt)

        # === HISTORY LOGGING ===
        seed_history[str(seed)] = {
            "train_acc":  r["log"]["train_acc"],
            "val_acc":    r["log"]["val_acc"],
            "train_loss": r["log"]["train_loss"],
            "val_loss":   r["log"]["val_loss"],
            "sparsity":   r["log"]["sparsity"],
        }

        sp   = r["stats"]["sparsity_pct"]
        ez   = r["stats"]["exact_zero_pct"]
        tc   = r["stats"]["true_compression"]
        b1   = r["bench"][1][0]
        b128 = r["bench"][128][0]
        print(f"  seed={seed:2d} | test={r['test_acc']:.2f}%  "
              f"sparsity={sp:.1f}%  exact0={ez:.1f}%  "
              f"true_compress={tc:.2f}x  "
              f"latency(bs=1)={b1:.2f}ms  latency(bs=128)={b128:.2f}ms")
        seed_results.append(r)

    test_accs   = [r["test_acc"]                      for r in seed_results]
    sparsities  = [r["stats"]["sparsity_pct"]         for r in seed_results]
    exact_zeros = [r["stats"]["exact_zero_pct"]       for r in seed_results]
    true_comps  = [r["stats"]["true_compression"]     for r in seed_results]
    head_comps  = [r["stats"]["headline_compression"] for r in seed_results]
    lat1s       = [r["bench"][1][0]                   for r in seed_results]
    lat128s     = [r["bench"][128][0]                 for r in seed_results]

    mean_acc = np.mean(test_accs)
    rep_idx  = int(np.argmin(np.abs(np.array(test_accs) - mean_acc)))
    rep      = seed_results[rep_idx]

    print(f"\n  MEAN +/- STD ({n_seeds} seeds)")
    print(f"    test_acc       = {np.mean(test_accs):.2f} +/- {np.std(test_accs):.2f}%")
    print(f"    sparsity       = {np.mean(sparsities):.1f} +/- {np.std(sparsities):.1f}%")
    print(f"    exact-zero     = {np.mean(exact_zeros):.1f} +/- {np.std(exact_zeros):.1f}%")
    print(f"    true compress  = {np.mean(true_comps):.2f} +/- {np.std(true_comps):.2f}x")
    print(f"    latency bs=1   = {np.mean(lat1s):.2f} +/- {np.std(lat1s):.2f} ms")
    print(f"    latency bs=128 = {np.mean(lat128s):.2f} +/- {np.std(lat128s):.2f} ms")

    return dict(
        model                = rep["model"],
        log                  = rep["log"],
        seed_history         = seed_history,
        test_acc             = float(np.mean(test_accs)),
        test_acc_std         = float(np.std(test_accs)),
        sparsity             = float(np.mean(sparsities)),
        sparsity_std         = float(np.std(sparsities)),
        exact_zero_pct       = float(np.mean(exact_zeros)),
        true_compression     = float(np.mean(true_comps)),
        true_compression_std = float(np.std(true_comps)),
        headline_compression = float(np.mean(head_comps)),
        latency_bs1          = float(np.mean(lat1s)),
        latency_bs1_std      = float(np.std(lat1s)),
        latency_bs128        = float(np.mean(lat128s)),
        latency_bs128_std    = float(np.std(lat128s)),
        stats                = rep["stats"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# 11.  Visualization — existing plots (unchanged)
# ═════════════════════════════════════════════════════════════════════════════

_COLORS = ["#9E9E9E", "#8BC34A", "#4CAF50", "#2196F3", "#FF9800", "#F44336"]


def plot_gate_histogram(model, lam, path):
    gates = model.all_gates().numpy()
    pct   = (gates < THRESHOLD).mean() * 100
    ez    = (gates == 0.0).mean() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gates, bins=100, color="#2196F3", edgecolor="white",
            linewidth=0.3, alpha=0.85)
    ax.axvline(THRESHOLD, color="#F44336", lw=2.0, ls="--",
               label=f"Threshold={THRESHOLD} ({pct:.1f}% pruned, {ez:.1f}% exactly 0)")
    ax.set_xlabel("Gate Value", fontsize=11)
    ax.set_ylabel("Count",      fontsize=11)
    ax.set_title(f"Gate Value Distribution  (lam={lam})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_combined_histograms(results, lambdas, path):
    n   = len(lambdas)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, lam, col in zip(axes, lambdas, _COLORS):
        gates  = results[lam]["model"].all_gates().numpy()
        pruned = (gates < THRESHOLD).mean() * 100
        ax.hist(gates, bins=80, color=col, edgecolor="white",
                linewidth=0.3, alpha=0.82)
        ax.axvline(THRESHOLD, color="#333", lw=1.2, ls="--")
        ax.set_title(f"lam={lam}\n({pruned:.1f}% pruned)", fontsize=9)
        ax.set_xlabel("Gate Value", fontsize=8)
        ax.set_ylabel("Count",      fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Gate Distributions Across Lambda Values", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_sparsity_vs_accuracy(results, lambdas, path):
    lam_nz   = [l for l in lambdas if l > 0]
    accs     = [results[l]["test_acc"]      for l in lam_nz]
    acc_std  = [results[l]["test_acc_std"]  for l in lam_nz]
    spars    = [results[l]["sparsity"]      for l in lam_nz]
    spar_std = [results[l]["sparsity_std"]  for l in lam_nz]
    tc       = [results[l]["true_compression"] for l in lam_nz]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    c1, c2, c3 = "#2196F3", "#F44336", "#4CAF50"

    ax1.errorbar(lam_nz, accs, yerr=acc_std, fmt="o-", color=c1,
                 lw=2.0, ms=8, capsize=4, label="Test Accuracy (%)")
    ax1.set_xscale("log")
    ax1.set_xlabel("Lambda (log scale)", fontsize=11)
    ax1.set_ylabel("Test Accuracy (%)", color=c1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=c1)

    ax2 = ax1.twinx()
    ax2.errorbar(lam_nz, spars, yerr=spar_std, fmt="s--", color=c2,
                 lw=2.0, ms=8, capsize=4, label="Sparsity (%)")
    ax2.set_ylabel("Sparsity (%)", color=c2, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=c2)

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.plot(lam_nz, tc, "^:", color=c3, lw=1.8, ms=7, label="True Compression (x)")
    ax3.set_ylabel("True Compression (x)", color=c3, fontsize=11)
    ax3.tick_params(axis="y", labelcolor=c3)

    lines  = [ax1.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0]]
    labels = ["Test Accuracy (%)", "Sparsity (%)", "True Compression (x)"]
    ax1.legend(lines, labels, loc="center left", fontsize=9)
    ax1.set_title("Sparsity vs. Accuracy vs. True Compression", fontsize=13)
    ax1.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_inference_speedup(results, lambdas, path):
    lam_labels = [str(l) for l in lambdas]
    lat1     = [results[l]["latency_bs1"]       for l in lambdas]
    lat128   = [results[l]["latency_bs128"]     for l in lambdas]
    lat1_s   = [results[l]["latency_bs1_std"]   for l in lambdas]
    lat128_s = [results[l]["latency_bs128_std"] for l in lambdas]

    x = np.arange(len(lambdas))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(x, lat1,   w, color="#2196F3")
    ax1.errorbar(x, lat1, yerr=lat1_s, fmt="none", color="black", capsize=3)
    ax1.set_xticks(x); ax1.set_xticklabels(lam_labels)
    ax1.set_title("Inference Latency — Batch Size 1", fontsize=11)
    ax1.set_xlabel("Lambda (0.0 = dense baseline)"); ax1.set_ylabel("Latency (ms)")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, lat128, w, color="#F44336")
    ax2.errorbar(x, lat128, yerr=lat128_s, fmt="none", color="black", capsize=3)
    ax2.set_xticks(x); ax2.set_xticklabels(lam_labels)
    ax2.set_title("Inference Latency — Batch Size 128", fontsize=11)
    ax2.set_xlabel("Lambda (0.0 = dense baseline)"); ax2.set_ylabel("Latency (ms)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Wall-Clock Inference Latency per Lambda", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# === PLOTTING ===
# New plotting functions that consume per-epoch history
# ═════════════════════════════════════════════════════════════════════════════

def _mean_across_seeds(seed_dict: dict, key: str) -> list:
    """Average a metric list across all seeds for one lambda."""
    arrays = [seed_dict[s][key] for s in seed_dict]
    return list(np.mean(arrays, axis=0))


def plot_training_curves(history: dict):
    """
    Per-epoch train_acc and val_acc averaged across seeds, one line per lambda.
    Saves: results/training_curves.png
    """
    # === PLOTTING ===
    lambdas = list(history.keys())
    colors  = _COLORS[:len(lambdas)]

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    for lam_str, col in zip(lambdas, colors):
        seed_dict  = history[lam_str]
        train_acc  = _mean_across_seeds(seed_dict, "train_acc")
        val_acc    = _mean_across_seeds(seed_dict, "val_acc")
        train_loss = _mean_across_seeds(seed_dict, "train_loss")
        val_loss   = _mean_across_seeds(seed_dict, "val_loss")
        epochs     = range(1, len(train_acc) + 1)

        ax_acc.plot(epochs, train_acc, color=col, lw=1.8, ls="-",
                    label=f"lam={lam_str} train")
        ax_acc.plot(epochs, val_acc,   color=col, lw=1.8, ls="--",
                    label=f"lam={lam_str} val")

        ax_loss.plot(epochs, train_loss, color=col, lw=1.8, ls="-",
                     label=f"lam={lam_str} train")
        ax_loss.plot(epochs, val_loss,   color=col, lw=1.8, ls="--",
                     label=f"lam={lam_str} val")

    ax_acc.set_title("Train & Validation Accuracy vs Epoch", fontsize=12)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend(fontsize=7, ncol=2); ax_acc.grid(True, alpha=0.25)

    ax_loss.set_title("Train & Validation Loss vs Epoch", fontsize=12)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7, ncol=2); ax_loss.grid(True, alpha=0.25)

    fig.suptitle("Training Curves per Lambda (mean across seeds)", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    plt.close()
    print("Plot    -> results/training_curves.png")


def plot_sparsity_curve(history: dict):
    """
    Per-epoch sparsity averaged across seeds, one line per lambda.
    Saves: results/sparsity_over_epochs.png
    """
    # === PLOTTING ===
    lambdas = list(history.keys())
    colors  = _COLORS[:len(lambdas)]

    fig, ax = plt.subplots(figsize=(9, 5))

    for lam_str, col in zip(lambdas, colors):
        seed_dict = history[lam_str]
        sparsity  = _mean_across_seeds(seed_dict, "sparsity")
        epochs    = range(1, len(sparsity) + 1)
        ax.plot(epochs, sparsity, color=col, lw=2.0, label=f"lam={lam_str}")

    ax.set_title("Sparsity Over Epochs per Lambda (mean across seeds)", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Sparsity (%)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig("results/sparsity_over_epochs.png", dpi=150)
    plt.close()
    print("Plot    -> results/sparsity_over_epochs.png")


# ═════════════════════════════════════════════════════════════════════════════
# 12.  Report Generator
# ═════════════════════════════════════════════════════════════════════════════

def generate_report(results: dict, lambdas: list) -> str:
    r0 = results[0.0]

    rows = []
    for lam in lambdas:
        r   = results[lam]
        tag = "  *(Baseline)*" if lam == 0.0 else ""
        rows.append(
            f"| {str(lam) + tag:<22} "
            f"| {r['test_acc']:>6.2f} ± {r['test_acc_std']:>4.2f}% "
            f"| {r['sparsity']:>5.1f} ± {r['sparsity_std']:>4.1f}% "
            f"| {r['exact_zero_pct']:>6.1f}% "
            f"| {r['true_compression']:>5.2f} ± {r['true_compression_std']:>4.2f}x "
            f"| {r['latency_bs1']:>6.2f} ms "
            f"| {r['latency_bs128']:>6.2f} ms |"
        )
    table = "\n".join(rows)

    r_low  = results[min([l for l in lambdas if l > 0], key=lambda x: abs(x - 0.1))]
    r_med  = results[min([l for l in lambdas if l > 0], key=lambda x: abs(x - 1.0))]
    r_high = results[max(lambdas)]
    baseline_lat1   = r0["latency_bs1"]
    baseline_lat128 = r0["latency_bs128"]

    report = f"""\
# Self-Pruning Neural Network — Experiment Report

**Method:** L0 regularisation with Hard Concrete gates
(Louizos, Welling & Kingma, ICLR 2018)

**Seeds:** {N_SEEDS} | **Epochs:** {NUM_EPOCHS} | **Warmup:** {WARMUP_EPOCHS} epochs
**Device:** {DEVICE}

---

## Results

| Lambda | Test Accuracy | Sparsity (%) | Exact Zero (%) | True Compression | Latency bs=1 | Latency bs=128 |
|--------|---------------|--------------|----------------|------------------|--------------|----------------|
{table}

> **True compression** = total_model_params / (active_prunable + non_prunable).

---

## Analysis

### λ = 0.0 — Baseline
- Accuracy: **{r0['test_acc']:.2f} ± {r0['test_acc_std']:.2f}%**
- Latency: **{baseline_lat1:.2f} ms** (bs=1), **{baseline_lat128:.2f} ms** (bs=128).

### λ ≈ 0.1 — Low pressure
- Accuracy: **{r_low['test_acc']:.2f} ± {r_low['test_acc_std']:.2f}%** (Δ = {r_low['test_acc'] - r0['test_acc']:+.2f}%)
- Sparsity: **{r_low['sparsity']:.1f} ± {r_low['sparsity_std']:.1f}%** (exact zero: {r_low['exact_zero_pct']:.1f}%)
- True compression: **{r_low['true_compression']:.2f}x**

### λ ≈ 1.0 — Medium pressure
- Accuracy: **{r_med['test_acc']:.2f} ± {r_med['test_acc_std']:.2f}%** (Δ = {r_med['test_acc'] - r0['test_acc']:+.2f}%)
- Sparsity: **{r_med['sparsity']:.1f} ± {r_med['sparsity_std']:.1f}%** (exact zero: {r_med['exact_zero_pct']:.1f}%)
- True compression: **{r_med['true_compression']:.2f}x**

### λ = {max(lambdas)} — High pressure
- Accuracy: **{r_high['test_acc']:.2f} ± {r_high['test_acc_std']:.2f}%** (Δ = {r_high['test_acc'] - r0['test_acc']:+.2f}%)
- Sparsity: **{r_high['sparsity']:.1f} ± {r_high['sparsity_std']:.1f}%**
- True compression: **{r_high['true_compression']:.2f}x**
"""
    return report


# ═════════════════════════════════════════════════════════════════════════════
# 13.  Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # === SAVE FILES ===
    os.makedirs("results", exist_ok=True)

    print(f"Device  : {DEVICE}")
    print(f"Seeds   : {N_SEEDS}")
    print(f"Epochs  : {NUM_EPOCHS}  (warmup: {WARMUP_EPOCHS})")
    print(f"Lambda  : {LAMBDAS}\n")

    train_dl, val_dl, test_dl = build_dataloaders(batch_size=128)
    results = {}

    # === HISTORY LOGGING ===
    # history[str(lam)][str(seed)] = {train_acc, val_acc, train_loss, val_loss, sparsity}
    history = {}

    for lam in LAMBDAS:
        results[lam] = run_experiment(
            lam, train_dl, val_dl, test_dl,
            num_epochs=NUM_EPOCHS, n_seeds=N_SEEDS,
        )

        # === HISTORY LOGGING ===
        history[str(lam)] = results[lam]["seed_history"]

        # === SAVE FILES ===
        # Save after each lambda — crash safety
        with open("results/history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"History -> results/history.json  (lam={lam} written)")

    # ── Summary table ─────────────────────────────────────────────────────
    bar = "=" * 110
    print(f"\n{bar}")
    print(f"{'lam':>6}  {'Acc':>12}  {'Sparsity':>12}  {'ExactZero':>10}  "
          f"{'TrueComp':>10}  {'Lat(bs1)':>10}  {'Lat(bs128)':>12}")
    print("-" * 110)
    for lam in LAMBDAS:
        r = results[lam]
        print(
            f"{lam:>6}  "
            f"{r['test_acc']:>6.2f}+/-{r['test_acc_std']:>4.2f}%  "
            f"{r['sparsity']:>5.1f}+/-{r['sparsity_std']:>4.1f}%  "
            f"{r['exact_zero_pct']:>9.1f}%  "
            f"{r['true_compression']:>9.2f}x  "
            f"{r['latency_bs1']:>9.2f}ms  "
            f"{r['latency_bs128']:>11.2f}ms"
        )
    print(bar)

    # === PLOTTING ===
    # New per-epoch history plots
    plot_training_curves(history)
    plot_sparsity_curve(history)

    # Existing plots (unchanged)
    for lam in LAMBDAS:
        plot_gate_histogram(results[lam]["model"], lam,
                            f"results/gates_lam_{lam}.png")

    plot_combined_histograms(results, LAMBDAS, "results/gates_combined.png")
    plot_sparsity_vs_accuracy(results, LAMBDAS, "results/sparsity_vs_accuracy.png")
    plot_inference_speedup(results, LAMBDAS,    "results/inference_speedup.png")

    # ── Report & JSON ─────────────────────────────────────────────────────
    report_text = generate_report(results, LAMBDAS)
    with open("results/report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Report  -> results/report.md")

    # === SAVE FILES ===
    summary = {}
    for lam in LAMBDAS:
        r = results[lam]
        summary[str(lam)] = {
            "test_accuracy_mean_%":    round(r["test_acc"],              4),
            "test_accuracy_std_%":     round(r["test_acc_std"],          4),
            "sparsity_mean_%":         round(r["sparsity"],              4),
            "sparsity_std_%":          round(r["sparsity_std"],          4),
            "exact_zero_%":            round(r["exact_zero_pct"],        4),
            "true_compression_mean":   round(r["true_compression"],      4),
            "true_compression_std":    round(r["true_compression_std"],  4),
            "headline_compression":    round(r["headline_compression"],  4),
            "prunable_total":          r["stats"]["prunable_total"],
            "prunable_active":         r["stats"]["prunable_active"],
            "total_model_params":      r["stats"]["total_params"],
            "latency_bs1_ms":          round(r["latency_bs1"],           3),
            "latency_bs1_std_ms":      round(r["latency_bs1_std"],       3),
            "latency_bs128_ms":        round(r["latency_bs128"],         3),
            "latency_bs128_std_ms":    round(r["latency_bs128_std"],     3),
            "n_seeds":                 N_SEEDS,
        }
    with open("results/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Summary -> results/summary.json")

    print("\nDone. Output files:")
    print("  results/summary.json")
    print("  results/history.json")
    print("  results/training_curves.png")
    print("  results/sparsity_over_epochs.png")
    print("  results/sparsity_vs_accuracy.png")
    print("  results/gates_*.png")
    print("  results/inference_speedup.png")
    print("  results/report.md")


if __name__ == "__main__":
    main()
