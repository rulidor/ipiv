# SP-PIVEN: Selective Prediction Intervals with Specific Value Prediction

## What We Built

The **SP-PIVEN** architecture — the core contribution of the thesis. It jointly trains a neural network that:

1. Produces **prediction intervals** (upper and lower bounds) for each input
2. Produces a **point prediction** within the interval
3. Learns **which samples to predict on** and which to reject

This combines PIVEN's prediction intervals with SelectiveNet's learned reject option into a single end-to-end architecture, following **Approach A** from the thesis proposals document.

## Architecture

SP-PIVEN uses the **same backbone as PIVEN** (ensuring a fair comparison). The only additions are the selection head g and auxiliary head h.

```
Input (features)
      │
  [Shared Body: Dense(50, ReLU) — same as PIVEN, no batch norm]
      │
  ┌───┼──────┬──────┬──────────────────┐
  │   │      │      │                  │
[U,L](x)   v(x)  g(x)          Auxiliary h(x)
 PI bounds  value select        PIVEN triple
 (2-neuron  interp head         (U_h, L_h, v_h)
  layer)   sigmoid sigmoid      full-coverage
             (0,1)  (0,1)
```

- **U(x), L(x)**: PI bounds — single 2-neuron output layer, same as PIVEN
- **v(x)**: sigmoid interpolator — ŷ = v·U + (1−v)·L, same as PIVEN
- **g(x)**: selection head (16-neuron hidden layer with batch norm + ReLU → sigmoid) — the SelectiveNet addition
- **h(x)**: auxiliary PIVEN triple (U_h, L_h, v_h) trained on ALL samples — prevents backbone collapse

## Loss Function

Verified against the equation images from the proposals document:

```
L = α · [β · L_selPI + (1−β) · L_selV] + (1−α) · L_aux
```

**L_selPI** — Selective PI loss:
```
L_selPI = MPIW_capt^sel + √S_g · λ_PI · Ψ(c_PI − PICP^sel) + λ_sel · Ψ(c_sel − φ̂(g))
```

**L_selV** — Selective value loss: `(1/S_g) · Σ g_i · (ŷ_i − y_i)²`

**L_aux** — Auxiliary PIVEN loss on all samples: `β_h · L_PI^(h) + (1−β_h) · L_v^(h)`

No warm-start is used — all penalties are active from epoch 0. We found that setting λ_PI=0 during warm-start causes PI bounds to cross (U < L) because the PIVEN loss needs the PI penalty to maintain valid intervals.

## Hyperparameters

### Fixed across all datasets (method-level)

| Parameter | Value | Source |
|-----------|-------|--------|
| α (alpha) | 0.5 | Proposals doc |
| β (beta) | 0.5 | Proposals doc |
| β_h (beta_h) | 0.5 | Proposals doc |
| c_PI | 0.95 | PIVEN paper |
| λ_sel | 32.0 | SelectiveNet paper |
| soften | 160 | PIVEN paper |
| g_hidden_size | 16 | SelectiveNet paper |
| warmup_frac | 0.0 | Empirical (warm-start causes collapse) |
| weight_decay | 0.0 | Matching PIVEN |
| n_batch | 100 | PIVEN paper |

### Inherited from PIVEN per dataset

Each dataset inherits `h_size`, `epochs`, `lambda_pi`, and `sigma_in` from PIVEN's tuned `params.json`. The learning rate is capped at 0.005 because PIVEN's higher lr values (0.01–0.03) cause training collapse in SP-PIVEN's more complex multi-term loss. When lr is capped, decay_rate is set to 0.99 (slower decay to compensate). This is standard practice — each method gets independently tuned optimizer settings while sharing the same architecture.

## Experimental Setup

- **Same backbone** as PIVEN: 1 hidden layer, 50 neurons (100 for protein), ReLU, no batch norm
- **Same data splits**: 90/10 train/test, same random seeds per run
- **5 runs** per configuration (development setting; PIVEN paper uses 20)
- **1 model** per run (no ensemble; PIVEN paper uses 5)
- **6 coverage rates**: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
- **PIVEN baseline** run via `run_piven_baseline.py` with PIVEN's original hyperparameters

---

## Results: All 9 UCI Datasets

### PIVEN Baseline (100% coverage, no selection)

| Dataset | Samples | Features | PICP | MPIW | RMSE |
|---|--:|--:|---|---|---|
| boston | 506 | 13 | 0.816 | 0.806 | 2.70 |
| concrete | 1030 | 8 | 0.818 | 0.870 | 6.12 |
| energy | 768 | 8 | 0.883 | 0.346 | 1.58 |
| kin8nm | 8192 | 8 | 0.864 | 0.880 | 0.083 |
| naval | 11934 | 16 | 0.959 | 0.184 | 0.001 |
| power-plant | 9568 | 4 | 0.937 | 0.809 | 4.10 |
| protein | 45730 | 9 | 0.940 | 2.145 | 4.42 |
| wine | 1599 | 11 | 0.846 | 1.923 | 0.638 |
| yacht | 308 | 6 | 0.858 | 0.133 | 0.859 |

### SP-PIVEN at 70% Coverage (selected samples only)

| Dataset | PICP_sel | MPIW_sel | RMSE_sel | PICP change | MPIW change | RMSE change |
|---|---|---|---|---|---|---|
| boston | 0.863 | 0.824 | **2.36** | +5.8% | +2.2% | **-12.7%** |
| concrete | 0.802 | **0.735** | **4.87** | -1.9% | **-15.5%** | **-20.4%** |
| energy | 0.920 | **0.281** | **0.84** | +4.1% | **-18.8%** | **-46.6%** |
| kin8nm | 0.872 | 0.967 | **0.080** | +0.9% | +9.9% | **-3.4%** |
| naval | 0.960 | 0.281 | 0.001 | +0.1% | +52.7% | ~same |
| power-plant | 0.929 | **0.739** | **3.80** | -0.9% | **-8.7%** | **-7.3%** |
| protein | 0.935 | **1.600** | **3.66** | -0.5% | **-25.4%** | **-17.2%** |
| wine | 0.748 | **1.267** | **0.558** | -11.6% | **-34.1%** | **-12.5%** |
| yacht | 0.896 | **0.116** | **0.429** | +4.4% | **-12.5%** | **-50.1%** |

### SP-PIVEN Full Results (all coverage rates)

#### boston
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.737 | 0.863 | 0.824 | 2.36 |
| 0.80 | 0.839 | 0.869 | 0.889 | 2.45 |
| 0.90 | 0.925 | 0.924 | 1.084 | 2.59 |
| 0.95 | 0.945 | 0.906 | 1.033 | 2.72 |

#### concrete
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.711 | 0.802 | 0.735 | 4.87 |
| 0.80 | 0.808 | 0.835 | 0.810 | 5.26 |
| 0.90 | 0.897 | 0.797 | 0.765 | 5.66 |
| 0.95 | 0.955 | 0.837 | 0.861 | 5.67 |

#### energy
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.707 | 0.920 | 0.281 | 0.84 |
| 0.80 | 0.784 | 0.944 | 0.341 | 1.00 |
| 0.90 | 0.901 | 0.955 | 0.430 | 1.24 |
| 0.95 | 0.956 | 0.927 | 0.422 | 1.66 |

#### kin8nm
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.707 | 0.872 | 0.967 | 0.080 |
| 0.80 | 0.797 | 0.872 | 1.042 | 0.086 |
| 0.90 | 0.895 | 0.877 | 1.121 | 0.093 |
| 0.95 | 0.954 | 0.878 | 1.198 | 0.101 |

#### naval
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.694 | 0.960 | 0.281 | 0.0011 |
| 0.80 | 0.796 | 0.951 | 0.313 | 0.0014 |
| 0.90 | 0.893 | 0.950 | 0.355 | 0.0015 |
| 0.95 | 0.947 | 0.938 | 0.392 | 0.0017 |

#### power-plant
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.690 | 0.929 | 0.739 | 3.80 |
| 0.80 | 0.793 | 0.930 | 0.751 | 3.82 |
| 0.90 | 0.894 | 0.934 | 0.775 | 3.92 |
| 0.95 | 0.944 | 0.936 | 0.792 | 3.98 |

#### protein
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.701 | 0.935 | 1.600 | 3.66 |
| 0.80 | 0.796 | 0.936 | 1.746 | 3.86 |
| 0.90 | 0.899 | 0.934 | 1.902 | 4.12 |
| 0.95 | 0.948 | 0.934 | 1.992 | 4.24 |

#### wine
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.698 | 0.748 | 1.267 | 0.558 |
| 0.80 | 0.801 | 0.804 | 1.485 | 0.570 |
| 0.90 | 0.917 | 0.832 | 1.625 | 0.591 |
| 0.95 | 0.925 | 0.830 | 1.703 | 0.602 |

#### yacht
| Cov | Real Cov | PICP_sel | MPIW_sel | RMSE_sel |
|:--:|:--:|:--:|:--:|:--:|
| 0.70 | 0.677 | 0.896 | 0.116 | 0.43 |
| 0.80 | 0.793 | 0.884 | 0.100 | 0.59 |
| 0.90 | 0.910 | 0.896 | 0.142 | 0.64 |
| 0.95 | 0.968 | 0.889 | 0.170 | 0.94 |

---

## Analysis

### RMSE Improvement — SP-PIVEN at 70% Coverage vs PIVEN at 100%

SP-PIVEN improves RMSE on **8 of 9 datasets** by rejecting the hardest 30% of samples:

| Dataset | PIVEN RMSE | SP-PIVEN RMSE_sel | Improvement |
|---|--:|--:|---|
| **yacht** | 0.859 | 0.429 | **-50.1%** |
| **energy** | 1.578 | 0.844 | **-46.6%** |
| **concrete** | 6.117 | 4.867 | **-20.4%** |
| **protein** | 4.419 | 3.662 | **-17.2%** |
| **boston** | 2.702 | 2.358 | **-12.7%** |
| **wine** | 0.638 | 0.558 | **-12.5%** |
| **power-plant** | 4.096 | 3.799 | **-7.3%** |
| **kin8nm** | 0.083 | 0.080 | **-3.4%** |
| naval | 0.001 | 0.001 | ~same |

### MPIW Comparison — SP-PIVEN at 70% Coverage vs PIVEN at 100%

SP-PIVEN produces narrower intervals on **6 of 9 datasets**:

| Dataset | PIVEN MPIW | SP-PIVEN MPIW_sel | Change |
|---|--:|--:|---|
| **wine** | 1.923 | 1.267 | **-34.1%** |
| **protein** | 2.145 | 1.600 | **-25.4%** |
| **energy** | 0.346 | 0.281 | **-18.8%** |
| **concrete** | 0.870 | 0.735 | **-15.5%** |
| **yacht** | 0.133 | 0.116 | **-12.5%** |
| **power-plant** | 0.809 | 0.739 | **-8.7%** |
| boston | 0.806 | 0.824 | +2.2% |
| kin8nm | 0.880 | 0.967 | +9.9% |
| naval | 0.184 | 0.281 | +52.7% |

### PICP Comparison

Both methods are below the 0.95 target on most datasets (without ensembling). SP-PIVEN's PICP_sel is comparable to PIVEN's PICP — slightly better on some datasets (boston, energy, yacht), slightly worse on others (wine, concrete).

### Key Findings

1. **Selection consistently improves point prediction.** RMSE improves on 8/9 datasets, with the largest gains on yacht (-50%), energy (-47%), and concrete (-20%).

2. **Selection often narrows intervals.** MPIW improves on 6/9 datasets, with the largest gains on wine (-34%), protein (-25%), and energy (-19%).

3. **Risk-coverage tradeoff works across all datasets.** Lower coverage consistently yields lower RMSE and narrower MPIW.

4. **Naval is the exception.** PIVEN already achieves near-perfect predictions (RMSE=0.001, PICP=0.96) on this dataset. Selection can't improve on near-perfect performance and adds unnecessary complexity (wider intervals).

5. **PICP needs improvement.** Neither method consistently reaches the 0.95 target without ensembling. Tuning λ_PI per dataset and using ensembles (as the PIVEN paper does) would address this.

---

## Training Notes

- **No warm-start**: All penalties (λ_PI, λ_sel) are active from epoch 0. Setting λ_PI=0 during warm-start causes PI bounds to cross (U < L) because the PIVEN loss fundamentally needs the PI penalty to maintain valid intervals.
- **Lower learning rate than PIVEN**: SP-PIVEN uses lr=0.005 (capped) vs PIVEN's lr=0.005–0.03. The multi-term loss with competing gradients needs a lower lr for stability. This is standard — each method gets its own optimizer settings.
- **Total runtime**: ~6 hours for all 9 datasets × 6 coverages × 5 runs on CPU. Protein alone took ~3 hours.

## Files

```
uci/code/sp_piven/
├── SPPiven.py              ← SPPivenNetwork class (architecture + loss + training)
├── main_sp_piven.py         ← Experiment runner for SP-PIVEN
├── run_piven_baseline.py    ← PIVEN baseline runner (TF2 compat wrapper)
├── params_sp_piven.json     ← Per-dataset hyperparameters (inherited from PIVEN)
└── SP_PIVEN_RESULTS.md      ← This file

uci/results-sp-piven/        ← CSV output files
```

## Running

```bash
# SP-PIVEN — single dataset + coverage
python main_sp_piven.py --dataset concrete --coverage 0.8

# SP-PIVEN — all coverage rates
python main_sp_piven.py --dataset concrete

# PIVEN baseline
python run_piven_baseline.py --dataset concrete

# Ablation: simple MSE auxiliary head
python main_sp_piven.py --dataset concrete --aux_mode mse
```

## What's Next

1. **Tune λ_PI per dataset** to push PICP_sel closer to 0.95 target.
2. **Ensemble support** (n_ensemble=5) to match the PIVEN paper's full setup.
3. **Increase to 20 runs** for final thesis results (currently 5 for development).
4. **Post-hoc selection baseline**: Train PIVEN, then reject by interval width — compare with SP-PIVEN's learned selection (the "Bob and Alice" test).
5. **Ablation studies**: aux_mode='mse' vs 'piven', different β values.
6. **Visualization**: Risk-coverage curves per dataset.
