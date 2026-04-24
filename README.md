# Self-Pruning Neural Network

> **A neural network that learns to eliminate its own unnecessary connections during training — no post-processing required.**

[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Run%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/megeshsundharaj/self-pruning-neural-network-tredance)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## Overview

This project implements **differentiable structured pruning** for CIFAR-10 image classification. Instead of pruning weights after training, the network learns — during training — which connections to keep and which to discard.

Each weight in every layer is paired with a learnable **gate parameter**. A sparsity penalty in the loss function pushes most gates toward exact zero, shrinking the network while preserving accuracy on the connections that matter.

**Key achievement:** At the recommended operating point (λ = 0.3), the model prunes **59.3 % of all weights to exact zero** with a **+0.15 % accuracy gain** over the dense baseline, achieving **2.46× true compression**.

---

## Live Demo

Run the full experiment end-to-end on Kaggle (no local GPU required):

**[Kaggle Notebook — Self-Pruning Neural Network](https://www.kaggle.com/code/megeshsundharaj/self-pruning-neural-network-tredance)**

---

## Architecture

### PrunableLinear Layer

A drop-in replacement for `nn.Linear` that wraps every weight with a learnable gate:

```
gate  =  sigmoid(gate_scores)          # ∈ (0, 1) per weight
w_eff =  weight × gate                 # element-wise gated weight
out   =  x @ w_eff.T + bias            # standard linear projection
```

- `weight` and `gate_scores` are both registered `nn.Parameter` tensors.
- The sigmoid keeps gates in (0, 1); a gate near 0 effectively zeros the weight.
- Gradients flow through both tensors — the network jointly optimises accuracy and sparsity.

### Network Architecture (SelfPruningNet)

```
Input (3 × 32 × 32)
 └─ Conv Block 1 :  PrunableConv2d(3 → 32)   + BN + ReLU + MaxPool
 └─ Conv Block 2 :  PrunableConv2d(32 → 64)  + BN + ReLU + MaxPool
 └─ Conv Block 3 :  PrunableConv2d(64 → 128) + BN + ReLU + MaxPool
 └─ Flatten → 2048
 └─ PrunableLinear(2048 → 512) + Dropout(0.4)
 └─ PrunableLinear(512 → 256)  + Dropout(0.3)
 └─ PrunableLinear(256 → 10)   → logits
```

### Loss Function

```
Total Loss = CrossEntropyLoss  +  λ × SparsityLoss
```

`SparsityLoss` is the **L1 norm of all gate values** (sum of sigmoid outputs across every PrunableLinear/PrunableConv2d layer). Because the L1 penalty applies a constant downward gradient regardless of gate magnitude, it drives low-activity gates all the way to zero — producing true structural sparsity.

---

## Results

All experiments run over **3 random seeds** (0, 1, 42), **30 epochs**, **5-epoch linear warmup** of λ.

| λ | Test Accuracy | Sparsity (%) | Exact Zero (%) | True Compression |
|---|:---:|:---:|:---:|:---:|
| 0.0 *(baseline)* | 79.01 ± 0.12 % |  0.0 % |  0.0 % | 1.00× |
| 0.1 | 79.14 ± 0.07 % | 54.3 ± 1.7 % | 53.6 % | 2.19× |
| **0.3** *(recommended)* | **79.16 ± 0.38 %** | **59.3 ± 1.9 %** | **59.1 %** | **2.46×** |
| 1.0 | 78.95 ± 0.26 % | 61.8 ± 1.8 % | 61.6 % | 2.62× |
| 2.0 | 77.93 ± 0.26 % | 67.1 ± 1.5 % | 66.6 % | 3.04× |
| 5.0 | 75.20 ± 0.21 % | 81.8 ± 0.3 % | 81.2 % | 5.47× |

> **λ = 0.3** is the sweet spot: 59 % of weights pruned, best accuracy in the sweep, 2.46× smaller model — at zero accuracy cost.

---

## Visualisations

| Plot | What it shows |
|------|---------------|
| `results/training_curves.png` | Train / val accuracy & loss per epoch across all λ |
| `results/sparsity_over_epochs.png` | How sparsity evolves during training |
| `results/sparsity_vs_accuracy.png` | Accuracy–sparsity–compression trade-off |
| `results/gates_lam_1.0.png` | Gate value distribution at λ = 1.0 (large spike at 0) |
| `results/gates_combined.png` | Gate distributions across all λ values |
| `results/inference_speedup.png` | Wall-clock latency at batch sizes 1 and 128 |

---

## Quickstart

### 1. Clone & install

```bash
git clone <repo-url>
cd self_pruning_nn
pip install -r requirements.txt
```

### 2. Train all λ configurations

```bash
python self_pruning_nn_final.py
```

The script will:
- Download CIFAR-10 automatically on first run
- Train across λ ∈ {0.0, 0.1, 0.3, 1.0, 2.0, 5.0} with 3 seeds each
- Save all plots to `results/`
- Print a Markdown summary table to stdout

### 3. Requirements

```
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.24.0
matplotlib >= 3.7.0
```

A CUDA-capable GPU is recommended; the code falls back to CPU automatically.

---

## Project Structure

```
self_pruning_nn/
├── self_pruning_nn_final.py   # All code: layers, network, training, evaluation
├── requirements.txt
├── README.md
├── REPORT.md                  # Method explanation, results analysis, figures index
└── results/
    ├── summary.json
    ├── history.json
    ├── training_curves.png
    ├── sparsity_over_epochs.png
    ├── sparsity_vs_accuracy.png
    ├── inference_speedup.png
    ├── gates_lam_1.0.png
    └── gates_combined.png
```

---

## How It Works — The Short Version

1. **Every weight gets a gate.** `gate_scores` (same shape as `weight`) are learned parameters. `gate = sigmoid(gate_scores)` squashes them to (0, 1).
2. **Gated weights replace real weights.** The forward pass uses `weight × gate` instead of `weight`.
3. **The loss penalises open gates.** `λ × Σ gates` adds a constant per-gate cost, pulling unused gates toward zero.
4. **Exact zeros emerge.** Because the gradient of the L1 term is constant (not proportional to gate size), the optimizer cannot find an equilibrium with small-but-nonzero gates for connections the task doesn't need — they collapse all the way to zero.

---

## Reference

Louizos, C., Welling, M., & Kingma, D. P. (2018).  
*Learning Sparse Neural Networks through L0 Regularization.*  
ICLR 2018. [arxiv.org/abs/1712.01312](https://arxiv.org/abs/1712.01312)

---

*Submitted as part of the Tredence Analytics AI Engineer Case Study.*
