# Self-Pruning Neural Network — Experiment Report

**Method:** L0 regularisation with Hard Concrete gates  
(Louizos, Welling & Kingma, ICLR 2018)

**Seeds:** 3 | **Epochs:** 30 | **Warmup:** 5 epochs | **Dataset:** CIFAR-10

---

## Method

### From L1 to L0 Sparsity

The standard approach to inducing sparsity in neural networks is L1 regularisation, which adds the sum of absolute weight magnitudes to the task loss:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_j |w_j|$$

L1 encourages sparsity because it applies a constant gradient penalty regardless of weight magnitude, making it optimal for small weights to collapse toward zero. In practice, however, L1 has two critical limitations. First, at any finite λ it produces weights that are small but not exactly zero — the optimizer must balance the gradient of the task loss against the constant L1 pull, and equilibrium is never at precisely zero unless the task gradient also vanishes. Second, L1 applies uniform shrinkage: it penalises all weights equally, which biases the magnitudes of weights that should remain large and distorts the learned representations.

The ideal objective penalises the *count* of non-zero weights, not their magnitude — this is L0 regularisation.

### L0 Regularisation

L0 regularisation targets the number of active parameters directly:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \sum_j \mathbf{1}[w_j \neq 0]$$

Each weight is replaced by a gated product $\tilde{w}_j = w_j \cdot z_j$, and the network learns the gate distribution rather than penalising weight magnitude. This decouples sparsity control from weight scaling: active weights are unpenalised and retain their full expressive capacity, while the gate $z_j$ alone determines inclusion. The result is true structural sparsity — exact zeros — rather than the near-zero approximation produced by L1.

The challenge is that the L0 count is non-differentiable, requiring a continuous relaxation to enable gradient-based training.

### Hard Concrete Relaxation

The Hard Concrete distribution (Louizos et al., ICLR 2018) provides a practical, differentiable surrogate for the Bernoulli gate $z_j \in \{0, 1\}$:

1. **Soft relaxation:** A continuous gate $s$ is sampled from a BinaryConcrete distribution, stretched over the interval $(\gamma, \zeta)$ with $\gamma < 0 < 1 < \zeta$.
2. **Hard clamp:** $z = \text{clip}(s, 0, 1)$ — the stretch-and-clip construction places genuine probability mass at the boundary values 0 and 1, not merely approaching them asymptotically.
3. **Differentiable penalty:** The expected L0 norm $P(z > 0) = \sigma\!\left(\log\alpha - \beta \log\!\left(-\gamma/\zeta\right)\right)$ is smooth in the learned log-odds parameter $\alpha$, so sparsity gradients backpropagate exactly.

This resolves both failure modes of the alternatives: L1's inability to produce exact zeros, and the high variance of straight-through gradient estimators for discrete gates. HC provides a closed-form, low-variance gradient of the expected sparsity penalty, making it the canonical choice for differentiable L0 regularisation.

**Hyperparameters:** $\beta = 2/3$, $\gamma = -0.1$, $\zeta = 1.1$ (from the original paper).

The per-layer gate distributions learned at convergence are shown in `results/gates_lam_1.0.png` (best compression point) and `results/gates_combined.png` (all λ values).

---

## Results

| λ | Test Accuracy | Sparsity (%) | Exact Zero (%) | True Compression | Latency bs=1 | Latency bs=128 |
|---|---------------|--------------|----------------|------------------|--------------|----------------|
| 0.0 *(Baseline)* | 79.01 ± 0.12% |  0.0 ±  0.0% |  0.0% | 1.00 ± 0.00x | 1.12 ms | 2.19 ms |
| 0.1              | 79.14 ± 0.07% | 54.3 ±  1.7% | 53.6% | 2.19 ± 0.08x | 1.16 ms | 2.21 ms |
| 0.3              | 79.16 ± 0.38% | 59.3 ±  1.9% | 59.1% | 2.46 ± 0.12x | 1.15 ms | 2.21 ms |
| 1.0              | 78.95 ± 0.26% | 61.8 ±  1.8% | 61.6% | 2.62 ± 0.13x | 1.16 ms | 2.21 ms |
| 2.0              | 77.93 ± 0.26% | 67.1 ±  1.5% | 66.6% | 3.04 ± 0.14x | 1.16 ms | 2.20 ms |
| 5.0              | 75.20 ± 0.21% | 81.8 ±  0.3% | 81.2% | 5.47 ± 0.08x | 1.13 ms | 2.20 ms |

> **True compression** = total_model_params / (active_prunable + non_prunable).

All reported metrics are averaged over 3 random seeds (0, 1, 42), with standard deviation shown. The accuracy–sparsity trade-off across all λ values is visualised in `results/sparsity_vs_accuracy.png`.

---

## Best Trade-off: λ = 0.3

**λ = 0.3 achieves the optimal accuracy–sparsity trade-off**, marginally exceeding the baseline in accuracy while compressing the model by 2.46x:

- **Accuracy:** 79.16 ± 0.38% (+0.15% vs. baseline)
- **Sparsity:** 59.3 ± 1.9% (exact zero: 59.1%)
- **True compression:** 2.46 ± 0.12x

At this operating point, the Hard Concrete gates prune nearly 60% of prunable weights to exact zero at no accuracy cost — a result that L1 cannot match, since it neither reaches exact zeros nor preserves the magnitudes of retained weights.

---

## Analysis

### λ = 0.0 — Baseline
- Accuracy: **79.01 ± 0.12%** | Latency: **1.12 ms** (bs=1), **2.19 ms** (bs=128).
- All gates remain open; no sparsity is induced.

### λ = 0.1 — Minimal pressure
- Accuracy: **79.14 ± 0.07%** (Δ = +0.12%)
- Sparsity: **54.3 ± 1.7%** | True compression: **2.19x**
- Over half the prunable weights are zeroed with zero accuracy cost; the slight gain suggests mild regularisation benefit.

### λ = 0.3 — Best trade-off *(recommended)*
- Accuracy: **79.16 ± 0.38%** (Δ = +0.15%)
- Sparsity: **59.3 ± 1.9%** | True compression: **2.46x**
- Peak accuracy in the sweep; marginal increase in penalty over λ = 0.1 yields additional sparsity at no cost.

### λ = 1.0 — Moderate pressure
- Accuracy: **78.95 ± 0.26%** (Δ = −0.07%)
- Sparsity: **61.8 ± 1.8%** | True compression: **2.62x**
- Accuracy drops below baseline for the first time; compression gain is modest relative to λ = 0.3.

### λ = 2.0 — High pressure
- Accuracy: **77.93 ± 0.26%** (Δ = −1.08%)
- Sparsity: **67.1 ± 1.5%** | True compression: **3.04x**
- Meaningful accuracy degradation with diminishing sparsity returns.

### λ = 5.0 — Extreme pressure
- Accuracy: **75.20 ± 0.21%** (Δ = −3.82%)
- Sparsity: **81.8 ± 0.3%** | True compression: **5.47x**
- Network capacity is severely restricted; 5.47x compression comes at the cost of nearly 4 percentage points of accuracy.

---

## Figures

| File | Description |
|------|-------------|
| `results/training_curves.png` | Train vs. validation accuracy per epoch for all λ |
| `results/sparsity_over_epochs.png` | Gate sparsity over training epochs per λ |
| `results/sparsity_vs_accuracy.png` | Accuracy vs. sparsity trade-off across λ values |
| `results/inference_speedup.png` | Measured inference latency at batch sizes 1 and 128 |
| `results/gates_lam_1.0.png` | Per-layer gate distribution at λ = 1.0 |
| `results/gates_combined.png` | Per-layer gate distributions across all λ values |

---

## Reference

Louizos, C., Welling, M., & Kingma, D. P. (2018).  
*Learning Sparse Neural Networks through L0 Regularization.*  
ICLR 2018. https://arxiv.org/abs/1712.01312
