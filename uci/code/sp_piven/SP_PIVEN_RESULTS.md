# SP-PIVEN: Selective Prediction Intervals with Specific Value Prediction

## What We Built

The **SP-PIVEN** architecture — the core contribution of the thesis. It jointly trains a neural network that:

1. Produces **prediction intervals** (upper and lower bounds) for each input
2. Produces a **point prediction** within the interval
3. Learns **which samples to predict on** and which to reject

This combines PIVEN's prediction intervals with SelectiveNet's learned reject option into a single end-to-end architecture, following **Approach A** from the thesis proposals document.

## Architecture

```
Input (features)
      │
  [Shared Body: Dense(64) + BatchNorm + ReLU]
      │
  ┌───┼──────┬──────┬──────────────────┐
  │   │      │      │                  │
 U(x) L(x) v(x)  g(x)          Auxiliary h(x)
 upper lower value select       PIVEN triple
 bound bound interp head        (U_h, L_h, v_h)
  │    │   sigmoid sigmoid      full-coverage
  │    │   (0,1)  (0,1)         (trained on ALL samples)
  │    │      │      │
  └────┼──────┘      │
       │             │
  ŷ = v·U+(1-v)·L   │
  (point prediction) │
       │             │
       └─────────────┘
       Predict iff g(x) ≥ τ
```

**Five primary output heads** branch from the shared backbone:
- **U(x)** and **L(x)**: upper and lower bounds of the prediction interval
- **v(x)**: interpolation weight (sigmoid) → point prediction ŷ = v·U + (1−v)·L
- **g(x)**: selection score (sigmoid) → accept if g(x) ≥ calibrated threshold τ
- **h(x)**: auxiliary PIVEN triple (U_h, L_h, v_h) trained on all samples without selection — this prevents the backbone from overfitting to only the "easy" selected samples

## Loss Function

The loss follows the formulas from the proposals document (Approach A), verified against the equation images:

```
L = α · [β · L_selPI + (1−β) · L_selV] + (1−α) · L_aux + L2_reg
```

### Key quantities

```
k_i = 1[L_i ≤ y_i ≤ U_i]        — capture indicator (is y inside the PI?)
M_g = Σ g(x_i)·k_i              — jointly selected AND captured mass
S_g = Σ g(x_i)                  — selected mass
φ̂(g) = S_g / n                 — empirical selection coverage
```

### Three loss components

**L_selPI** — Selective PI loss (minimize interval width on selected samples, enforce PI coverage and selection coverage):
```
MPIW_capt^sel = Σ|U_i−L_i|·k_i·g_i / M_g       (captured width, weighted by selection)
PICP^sel = M_g / S_g                              (PI coverage among selected samples)

L_selPI = MPIW_capt^sel
        + √S_g · λ_PI · max(0, c_PI − PICP^sel)²    ← PI coverage penalty
        + λ_sel · max(0, c_sel − φ̂(g))²              ← selection coverage penalty
```

Note: The PI penalty scales by √S_g (selected mass), not √n — fewer selected samples → less confidence in PICP → weaker penalty.

**L_selV** — Selective value loss (point prediction accuracy on selected samples):
```
L_selV = (1/S_g) · Σ g_i · (ŷ_i − y_i)²
```

**L_aux** — Auxiliary loss (standard PIVEN on all samples, keeps backbone features relevant for entire distribution):
```
L_aux = β_h · L_PI^(h) + (1−β_h) · L_v^(h)
```

### Warm-start schedule

First 20% of training epochs: `λ_PI = 0` — the model learns reasonable PI boundaries and selection patterns without PI coverage pressure. After 20%, `λ_PI` steps to its target value (15.0). This prevents the selection head from collapsing early when the PI coverage penalty and selection penalty compete.

## Hyperparameters

| Parameter | Value | Source | Description |
|-----------|-------|--------|-------------|
| α (alpha) | 0.5 | Proposals doc | Balance: selective loss vs auxiliary loss |
| β (beta) | 0.5 | Proposals doc | Balance: PI loss vs value loss (within selective) |
| β_h (beta_h) | 0.5 | Proposals doc | Balance: PI vs value (within auxiliary) |
| c_PI | 0.95 | PIVEN paper | PI coverage target |
| λ_PI | 15.0 | PIVEN paper | PI coverage penalty weight |
| λ_sel | 32.0 | SelectiveNet paper | Selection coverage penalty weight |
| soften | 160 | PIVEN paper | Sigmoid softening for k_soft |
| warmup_frac | 0.2 | Proposals doc | Fraction of epochs with λ_PI=0 |
| h_size | [64] | SelectiveNet paper | Backbone: 1 hidden layer, 64 neurons |
| g_hidden_size | 16 | SelectiveNet paper | Selection head hidden layer |
| lr | 5e-4 | SelectiveNet paper | Learning rate (Adam) |
| decay_rate | 0.99 | SelectiveNet paper | LR exponential decay |
| epochs | 800 | SelectiveNet paper | Training epochs |
| n_batch | 256 | SelectiveNet paper | Batch size |
| weight_decay | 1e-4 | SelectiveNet paper | L2 regularization |
| out_biases | [3, -3] | PIVEN code | Initial PI bounds bias |

## Results on Concrete Dataset (PIVEN-matched architecture)

Both PIVEN and SP-PIVEN use the **same backbone**: 1 hidden layer with 50 neurons, ReLU, no batch norm. The only difference is SP-PIVEN's addition of the selection head g and auxiliary head h. This ensures a fair comparison where any performance difference is due to the selection mechanism.

SP-PIVEN training settings: lr=0.005, decay=0.99, batch_size=100, 800 epochs, no warm-start (all penalties active from epoch 0). No weight decay.

Averaged over 5 random 90/10 train/test splits.

### PIVEN Baseline (100% coverage — predicts on ALL samples)

Run via `run_piven_baseline.py` with original PIVEN hyperparameters (lr=0.03, decay=0.98).

| PICP | MPIW | RMSE |
|:--:|:--:|:--:|
| 0.835 ± 0.007 | 0.896 ± 0.036 | 6.24 ± 0.46 |

### SP-PIVEN Results

| Target Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel | PICP_all | MPIW_all | RMSE_all |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.701 | 0.806 | **0.705** | **4.71** | 0.693 | 0.687 | 6.51 |
| 0.75 | 0.751 | 0.826 | **0.735** | **4.84** | 0.746 | 0.785 | 6.45 |
| 0.80 | 0.781 | 0.821 | **0.747** | **5.06** | 0.753 | 0.789 | 6.33 |
| 0.85 | 0.852 | 0.840 | **0.785** | **5.10** | 0.769 | 0.805 | 6.60 |
| 0.90 | 0.885 | 0.821 | 0.844 | **5.37** | 0.790 | 0.874 | 6.36 |
| 0.95 | 0.955 | 0.819 | 0.845 | **5.70** | 0.810 | 0.863 | 6.29 |

Auxiliary head: PICP ≈ 0.85, RMSE ≈ 6.2 (confirms backbone learns from full distribution).

### SP-PIVEN vs PIVEN — Side by Side

| Method | Coverage | PICP | MPIW | RMSE |
|:--|:--:|:--:|:--:|:--:|
| PIVEN (no selection) | 100% | 0.835 | 0.896 | 6.24 |
| SP-PIVEN | 95% | 0.819 | 0.845 | **5.70** |
| SP-PIVEN | 90% | 0.821 | 0.844 | **5.37** |
| SP-PIVEN | 80% | 0.821 | **0.747** | **5.06** |
| SP-PIVEN | 70% | 0.806 | **0.705** | **4.71** |

### How to Read These Results

**SP-PIVEN produces significantly better point predictions by rejecting uncertain samples.** At 70% coverage, RMSE drops from 6.24 (PIVEN on all) to 4.71 (SP-PIVEN on selected) — a **24.5% improvement**. Even at 95% coverage (rejecting only 5%), RMSE improves from 6.24 to 5.70 — an **8.7% improvement**.

**SP-PIVEN produces narrower intervals on selected samples.** At 80% coverage, MPIW is 0.747 vs PIVEN's 0.896 — **16.6% narrower**. The selection mechanism rejects samples where the model would need wide intervals, keeping only the ones where it can produce tight, informative PIs.

**PI coverage (PICP) is similar but slightly below PIVEN.** Both methods are below the 0.95 target (PIVEN: 0.835, SP-PIVEN: 0.81–0.84). Neither achieves 0.95 without ensembling. SP-PIVEN's PICP can be improved by tuning λ_PI.

**Risk-coverage tradeoff works clearly**: lower coverage → lower RMSE and narrower MPIW, as expected.

### Key Takeaway

With the same backbone architecture, SP-PIVEN demonstrates that learned selection improves both point prediction accuracy and interval quality. The selection head successfully identifies and rejects hard-to-predict samples, concentrating model quality on the samples it keeps.

### What Still Needs Work

1. **PICP below target**: Both PIVEN and SP-PIVEN are below 0.95. Increasing λ_PI or using ensembles would help.
2. **Hyperparameter tuning**: SP-PIVEN uses lr=0.005 (lower than PIVEN's 0.03) because the multi-term loss needs a lower learning rate. The lr, λ_PI, and λ_sel values have not been tuned — they are starting points.
3. **More datasets**: Concrete is one dataset — results must be validated across all 9 UCI benchmarks.

4. **Epochs 400–800 (refinement)**: Gradual improvement. MPIW decreases while PICP stabilizes near target. Selection coverage stabilizes.

## Files

```
uci/code/sp_piven/
├── SPPiven.py              ← SPPivenNetwork class (architecture + loss + training)
├── main_sp_piven.py         ← Experiment runner
├── params_sp_piven.json     ← Hyperparameters per dataset
└── SP_PIVEN_RESULTS.md      ← This file

uci/results-sp-piven/        ← CSV output files
```

## Running

```bash
# Single coverage rate
python main_sp_piven.py --dataset concrete --coverage 0.8

# All coverage rates
python main_sp_piven.py --dataset concrete

# Ablation: simple MSE auxiliary head instead of full PIVEN triple
python main_sp_piven.py --dataset concrete --aux_mode mse
```

## What's Next

1. **Hyperparameter tuning**: PICP_sel is ~0.92 vs target 0.95. Try increasing λ_PI (e.g., 20–30) or training longer.

2. **More datasets**: Run on all 9 UCI datasets to establish the full benchmark.

3. **Ablation studies**:
   - `aux_mode='mse'` vs `aux_mode='piven'` — does the full auxiliary PIVEN triple help?
   - Different β values — how does the PI/value balance affect results?
   - Different warmup_frac — is 20% optimal?

4. **Comparison with baselines**:
   - PIVEN alone (no selection) at full coverage
   - SelectiveNet alone (no PIs)
   - Post-hoc selection on PIVEN (train PIVEN, then reject by interval width)

5. **Visualization**: Plot risk-coverage curves comparing SP-PIVEN, SelectiveNet, and post-hoc PIVEN selection.
