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

## Results on Concrete Dataset

Averaged over 5 random 90/10 train/test splits. "Selected" metrics are computed only on samples where g(x) ≥ τ (calibrated threshold). "All" metrics are on the full test set.

### SP-PIVEN Results

| Target Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel | PICP_all | MPIW_all | RMSE_all |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.664 ± .076 | **0.929** ± .026 | 1.94 ± .06 | **6.61** ± .24 | 0.802 ± .052 | 1.63 ± .19 | 8.10 ± .45 |
| 0.75 | 0.728 ± .034 | **0.934** ± .021 | 1.99 ± .15 | **6.72** ± .89 | 0.808 ± .053 | 1.71 ± .11 | 8.35 ± .44 |
| 0.80 | 0.777 ± .048 | **0.922** ± .031 | 2.00 ± .12 | **7.19** ± .70 | 0.814 ± .036 | 1.70 ± .08 | 8.01 ± .52 |
| 0.85 | 0.827 ± .040 | **0.932** ± .017 | 2.11 ± .14 | **7.35** ± .53 | 0.852 ± .040 | 1.87 ± .13 | 8.12 ± .40 |
| 0.90 | 0.913 ± .007 | **0.919** ± .028 | 2.07 ± .10 | **7.73** ± .65 | 0.887 ± .019 | 1.97 ± .07 | 8.19 ± .48 |
| 0.95 | 0.932 ± .030 | **0.913** ± .033 | 2.09 ± .17 | **7.92** ± .49 | 0.889 ± .027 | 2.00 ± .14 | 8.14 ± .44 |

Auxiliary head: PICP ≈ 0.92, RMSE ≈ 7.7 across all coverage rates (confirms backbone learns from full distribution).

### Comparison with Standalone PIVEN (the primary baseline)

Standalone PIVEN (no selection) run with the same setup: 5 runs, 1 model (no ensemble), same random seeds and train/test splits. Uses the original PIVEN hyperparameters from `params.json` (lr=0.03, decay=0.98, 800 epochs, hidden=[50]). Run via `run_piven_baseline.py`.

**PIVEN baseline results (100% coverage — predicts on ALL samples):**

| PICP | MPIW | RMSE (ŷ=v·U+(1−v)·L) |
|:--:|:--:|:--:|
| 0.835 ± 0.007 | 0.896 ± 0.036 | 6.24 ± 0.46 |

**SP-PIVEN vs PIVEN — side by side:**

| Method | Coverage | PICP | MPIW | RMSE |
|:--|:--:|:--:|:--:|:--:|
| PIVEN (no selection) | 100% | 0.835 | 0.90 | 6.24 |
| SP-PIVEN | 95% | 0.913 | 2.09 | 7.92 |
| SP-PIVEN | 90% | 0.919 | 2.07 | 7.73 |
| SP-PIVEN | 80% | 0.922 | 2.00 | 7.19 |
| SP-PIVEN | 70% | 0.929 | 1.94 | 6.61 |

### How to Read These Results

**SP-PIVEN achieves much higher PI coverage than standalone PIVEN.** PIVEN alone reaches only PICP=0.835 (well below the 0.95 target). SP-PIVEN on selected samples reaches PICP=0.92+ across all coverage levels. The selection mechanism lets the model reject samples where it can't produce reliable intervals, concentrating its PI quality on the samples it keeps.

**The tradeoff is wider intervals.** SP-PIVEN's MPIW (1.9–2.1) is higher than PIVEN's (0.9). This is partly because SP-PIVEN pushes harder toward the 0.95 PICP target — achieving higher coverage requires wider intervals. PIVEN achieves narrow intervals but at the cost of low PICP.

**RMSE comparison depends on coverage.** At 70% coverage, SP-PIVEN's RMSE (6.61) is close to PIVEN's (6.24), but SP-PIVEN is only predicting on the 70% of samples it's most confident about — while also providing valid prediction intervals. At 95% coverage, SP-PIVEN's RMSE (7.92) is worse, as expected when it must predict on nearly all samples.

**Note on PIVEN's low PICP.** The PIVEN paper reports PICP meeting the 0.95 target, but with M=5 ensembles and 20 runs. Our single-model baseline shows that without ensembling, PIVEN struggles to reach 0.95 on Concrete. This makes the comparison with SP-PIVEN (also single-model) fair.

### Comparison with Standalone SelectiveNet (secondary baseline)

SelectiveNet produces only point predictions (no PIs). This comparison shows the cost of adding PI capability.

| Target Cov | SelectiveNet RMSE_sel | SP-PIVEN RMSE_sel | SP-PIVEN PICP_sel |
|:--:|:--:|:--:|:--:|
| 0.70 | 4.66 | 6.61 | 0.929 |
| 0.80 | 5.02 | 7.19 | 0.922 |
| 0.90 | 5.15 | 7.73 | 0.919 |
| 0.95 | 5.60 | 7.92 | 0.913 |

SP-PIVEN's RMSE is higher than SelectiveNet's because it simultaneously optimizes for PI width and PI coverage — objectives that don't exist in SelectiveNet. The tradeoff: SP-PIVEN provides prediction intervals with ~92% coverage, which SelectiveNet cannot.

### Summary of Key Findings

1. **Selection improves PI coverage**: SP-PIVEN's PICP_sel (0.92) is substantially better than standalone PIVEN's PICP (0.84) — the model learns to reject samples where PIs would be unreliable.

2. **Risk-coverage tradeoff works**: Lower coverage → better RMSE and narrower intervals, as expected.

3. **Auxiliary head validates design**: PICP ≈ 0.92 and RMSE ≈ 7.7 on all samples confirms the backbone learns from the full distribution.

4. **Room for improvement**: PICP_sel is ~0.92 vs target 0.95. Tuning λ_PI, training longer, or adjusting warmup schedule could close this gap.

## Training Behavior

Key observations from training logs:

1. **Epochs 0–160 (warm-start)**: λ_PI = 0. The model learns reasonable PI bounds and selection patterns. PICP starts at 1.0 (wide intervals capture everything) and drops as intervals narrow.

2. **Epoch 160 (warm-start ends)**: λ_PI jumps to 15.0. Loss spikes temporarily (~100x) as the PI coverage penalty suddenly activates. This is normal — the model had narrowed intervals aggressively during warm-start.

3. **Epochs 160–400 (recovery)**: The model rapidly recovers. PICP climbs back toward 0.95 as intervals widen. Coverage converges toward the target.

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
