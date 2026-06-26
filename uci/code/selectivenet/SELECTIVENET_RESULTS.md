# SelectiveNet for Regression — Implementation & Results

## What We Built

A standalone implementation of **SelectiveNet** for regression tasks, following the architecture described in:

> *SelectiveNet: A Deep Neural Network with an Integrated Reject Option* (Geifman & El-Yaniv, ICML 2019)

This is a stepping stone toward **SP-PIVEN** (our thesis contribution), which will combine SelectiveNet's selection mechanism with PIVEN's prediction intervals.

## The Idea in Brief

A standard regression model must predict on every input — even when it's uncertain. SelectiveNet adds a **learned reject option**: the model decides which samples it's confident enough to predict on, and abstains on the rest.

The key insight is that learning to reject **end-to-end** (jointly with prediction) produces better results than applying a post-hoc confidence threshold to a pre-trained model. This is the "Bob and Alice" intuition from the paper: if you know in advance you only need to answer 5 out of 10 questions, you can study those 5 more deeply.

## Architecture

```
Input (x_size features)
       │
   ┌───┴───┐
   │ Main  │  ← Shared backbone: Dense(64) + BatchNorm + ReLU
   │ Body  │
   └───┬───┘
       │
   ┌───┼───────────────┐
   │   │               │
   ▼   ▼               ▼
  f(x) g(x)           h(x)
  Pred Selection      Auxiliary
  Head Head           Head
   │    │               │
   │   Dense(16)+BN    │
   │   +ReLU           │
   │    │               │
   │   Sigmoid          │
   │   (0 to 1)        │
   ▼    ▼               ▼
  1 linear  1 sigmoid  1 linear
  neuron    neuron     neuron
```

Three output heads branch from the shared body:

- **f(x)** — prediction head: outputs the regression value (linear activation)
- **g(x)** — selection head: outputs a score in (0, 1) indicating confidence. At inference, samples with g(x) ≥ τ are accepted; the rest are rejected.
- **h(x)** — auxiliary head: trained on the same regression task as f, but on **all** samples (no selection). This prevents the backbone from overfitting to only the "easy" selected samples.

## Loss Function

```
L = α · L_(f,g) + (1 − α) · L_h + L2_regularization
```

Where:
- **L_(f,g)** is the selective loss:
  ```
  L_(f,g) = selective_risk + λ · max(0, c − coverage)²
  ```
  - `selective_risk = Σ(MSE_i · g_i) / Σ(g_i)` — the weighted MSE, where g(x) acts as the weight. Samples the model "wants to reject" (low g) contribute less to the loss.
  - `λ · max(0, c − coverage)²` — a penalty that kicks in when the model rejects too many samples (coverage drops below target c). Without this, the model would trivially reject all hard samples.

- **L_h** is standard MSE on all samples — keeps the backbone learning useful features for the entire distribution.

- **α = 0.5** balances the two losses equally (same as in the paper).

## Hyperparameters

Following the SelectiveNet paper's regression setup:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| h_size | [64] | Main body: 1 hidden layer, 64 neurons |
| g_hidden_size | 16 | Selection head: hidden layer with 16 neurons |
| lambda_sel | 32 | Coverage penalty weight |
| alpha | 0.5 | Balance between selective and auxiliary loss |
| lr | 5e-4 | Learning rate (Adam optimizer) |
| decay_rate | 0.99 | Exponential learning rate decay |
| epochs | 800 | Training epochs |
| n_batch | 256 | Batch size |
| weight_decay | 1e-4 | L2 regularization |

## Post-Training Calibration

During training, the selection head g(x) outputs soft values in (0, 1). To achieve a precise target coverage at test time, we use **post-training calibration** (Section 5 of the paper):

1. Run the trained model on a held-out validation set
2. Collect all g(x) values
3. Set threshold τ = the (1 − c) percentile of g values
4. At test time: accept sample if g(x) ≥ τ, reject otherwise

This ensures that exactly the top-c fraction of samples (by confidence) are accepted.

## Results on Concrete Dataset

The Concrete Compressive Strength dataset (1030 samples, 8 features) is the regression benchmark used in the SelectiveNet paper. Results averaged over 5 random 90/10 train/test splits:

| Target Coverage | Realized Coverage | Selective RMSE | Full RMSE | Aux RMSE |
|:-:|:-:|:-:|:-:|:-:|
| 0.70 | 0.660 ± 0.050 | 4.66 ± 0.27 | 6.00 ± 0.20 | 6.02 ± 0.23 |
| 0.75 | 0.711 ± 0.040 | 5.02 ± 0.52 | 6.06 ± 0.23 | 5.99 ± 0.23 |
| 0.80 | 0.775 ± 0.045 | 5.02 ± 0.59 | 6.03 ± 0.16 | 6.08 ± 0.28 |
| 0.85 | 0.816 ± 0.047 | 5.19 ± 0.61 | 6.00 ± 0.28 | 6.01 ± 0.29 |
| 0.90 | 0.878 ± 0.020 | 5.15 ± 0.48 | 5.85 ± 0.18 | 5.91 ± 0.31 |
| 0.95 | 0.928 ± 0.011 | 5.60 ± 0.09 | 6.04 ± 0.20 | 6.04 ± 0.22 |

### How to Read This Table

- **Target Coverage**: how much of the test data we *want* the model to predict on (e.g., 0.70 = predict on 70%, reject 30%)
- **Realized Coverage**: how much it *actually* predicts on after calibration (close to target = good)
- **Selective RMSE**: prediction error on the *selected* (accepted) samples only — this is the metric we care about
- **Full RMSE**: prediction error on *all* samples (for comparison — what a standard model would achieve)
- **Aux RMSE**: error of the auxiliary head (should be similar to Full RMSE — confirms the backbone learns from all data)

### Key Observations

1. **Selection works**: Selective RMSE is consistently lower than Full RMSE across all coverage rates. At 70% coverage, the model achieves RMSE of 4.66 vs 6.00 on all samples — a **22% improvement** by rejecting the hardest 30% of samples.

2. **Risk-coverage tradeoff**: As coverage decreases (more rejection allowed), Selective RMSE improves. This is the expected behavior — the model gets to "cherry-pick" the easiest samples.

3. **Coverage calibration is close but not perfect**: Realized coverage tends to be slightly below target (e.g., 0.660 vs 0.700). This is a known issue discussed in the paper — small datasets lead to calibration variance.

4. **Auxiliary head confirms backbone quality**: Aux RMSE ≈ Full RMSE, meaning the auxiliary head successfully forces the backbone to learn features relevant to the entire dataset, not just the selected subset.

5. **Total runtime**: ~2.8 minutes for all 6 coverage rates × 5 runs = 30 experiments on CPU.

## Files

```
uci/code/selectivenet/
├── SelectiveNet.py              ← Network class (architecture + loss + training)
├── main_selectivenet.py         ← Experiment runner
├── params_selectivenet.json     ← Hyperparameters per dataset
└── SELECTIVENET_RESULTS.md      ← This file

uci/results-selectivenet/        ← CSV output files
```

## Running

```bash
# Single coverage rate
python main_selectivenet.py --dataset concrete --coverage 0.8

# All coverage rates
python main_selectivenet.py --dataset concrete

# Other datasets (same hyperparams, may need tuning)
python main_selectivenet.py --dataset boston
python main_selectivenet.py --dataset energy
```

## What's Next

This standalone SelectiveNet validates that the selection mechanism works for regression on UCI data. The next step is to **combine it with PIVEN** into the SP-PIVEN architecture (Approach A from the proposals document), where:
- The prediction head f(x) is replaced by PIVEN's three heads: U(x), L(x), v(x)
- The selection head g(x) remains the same
- The auxiliary head h(x) becomes a full PIVEN triple (U_h, L_h, v_h)
- The loss combines selective PI loss, selective value loss, auxiliary PIVEN loss, and coverage constraint
