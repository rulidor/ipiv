# -*- coding: utf-8 -*-
"""
Run standalone PIVEN as a baseline for comparison with SP-PIVEN.

This is a wrapper around the original PIVEN code (DeepNetPI.py / main.py)
that makes it compatible with our TF2 environment and matches the SP-PIVEN
experimental setup (same splits, seeds, and number of runs).

We don't modify the original PIVEN files — instead, we patch `tensorflow`
in sys.modules so that `import tensorflow as tf` in DeepNetPI.py resolves
to `tensorflow.compat.v1` with eager mode disabled.

Run as:
    python run_piven_baseline.py --dataset concrete
    python run_piven_baseline.py --dataset concrete --n_runs 5
"""
import sys
import os
import argparse
import json
import datetime
import numpy as np

# --- TF2 compat patch ---
# The original PIVEN code does `import tensorflow as tf` expecting TF1.
# We redirect this to tensorflow.compat.v1 so it works under TF2.
import tensorflow.compat.v1 as tf_compat
tf_compat.disable_eager_execution()
sys.modules['tensorflow'] = tf_compat

import scipy.stats as stats
import pandas as pd

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(parent_dir)
os.chdir(parent_dir)

from DataGen import DataGenerator
from DeepNetPI import TfNetwork
from utils import pi_to_gauss, gauss_neg_log_like, np_QD_loss
from sklearn.model_selection import train_test_split

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='concrete',
                    help='dataset name, from UCI_Datasets folder')
parser.add_argument('--n_runs', type=int, default=5,
                    help='number of runs (default 5 to match SP-PIVEN)')
args = parser.parse_args()

# Load PIVEN's original params
params_path = os.path.join(parent_dir, 'params.json')
with open(params_path) as f:
    all_params = json.load(f)
    try:
        params = next(el for el in all_params if el['dataset'] == args.dataset)
    except StopIteration:
        raise ValueError(f"Invalid dataset name: {args.dataset}")

n_runs = args.n_runs
n_epoch = params['epochs']
h_size = params['h_size']
l_rate = params['lr']
decay_rate = params['decay_rate']
soften = params['soften']
lambda_in = params['lambda_in']
sigma_in = params['sigma_in']
patience = params['patience']
alpha = 0.05
train_prop = 0.9
n_ensemble = 1  # no ensembling, to match SP-PIVEN setup
is_early_stop = patience != -1

if args.dataset == 'YearPredictionMSD':
    n_batch = 1000
    out_biases = [5., -5.]
else:
    n_batch = 100
    out_biases = [3., -3.]

print(f'\n{"="*70}')
print(f'PIVEN Baseline — {args.dataset} — {n_runs} runs, {n_ensemble} ensemble, {n_epoch} epochs')
print(f'{"="*70}')

results_runs = []
for run in range(n_runs):
    print(f'\n--- Run {run + 1}/{n_runs} ---')

    Gen = DataGenerator(dataset_name=args.dataset)
    X_train, y_train, X_test, y_test = Gen.create_data(
        seed_in=run, train_prop=train_prop)

    # Match SP-PIVEN: split off 20% validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=run)

    y_pred_all = []
    for i in range(n_ensemble):
        tf_compat.reset_default_graph()
        sess = tf_compat.Session()

        NN = TfNetwork(x_size=X_train.shape[1],
                       y_size=2,
                       h_size=h_size,
                       alpha=alpha,
                       soften=soften,
                       lambda_in=lambda_in,
                       sigma_in=sigma_in,
                       out_biases=out_biases,
                       method='piven',
                       patience=patience,
                       dataset=args.dataset)

        NN.train(sess, X_train, y_train, X_val, y_val,
                 n_epoch=n_epoch,
                 l_rate=l_rate,
                 decay_rate=decay_rate,
                 is_early_stop=is_early_stop,
                 n_batch=n_batch)

        y_loss, y_pred = NN.predict(sess, X_test=X_test, y_test=y_test)
        y_pred_all.append(y_pred)
        sess.close()

    y_pred_all = np.array(y_pred_all)

    # Extract predictions (single ensemble member or aggregate)
    if n_ensemble == 1:
        y_pred_U = y_pred_all[0, :, 0]
        y_pred_L = y_pred_all[0, :, 1]
        y_pred_v = y_pred_all[0, :, 2]
    else:
        _, _, y_pred_U, y_pred_L, y_pred_v = pi_to_gauss(y_pred_all, method='piven')

    y_true = y_test[:, 0]

    # PI metrics
    y_U_cap = y_pred_U > y_true
    y_L_cap = y_pred_L < y_true
    y_all_cap = y_U_cap * y_L_cap
    PICP = np.sum(y_all_cap) / len(y_true)
    MPIW = np.mean(y_pred_U - y_pred_L)

    # Point prediction (PIVEN: v*U + (1-v)*L)
    y_piven = y_pred_v * y_pred_U + (1 - y_pred_v) * y_pred_L
    RMSE = np.sqrt(np.mean(np.square(Gen.scale_c * (y_piven - y_true))))

    # Midpoint prediction (for reference)
    y_mid = 0.5 * y_pred_U + 0.5 * y_pred_L
    RMSE_mid = np.sqrt(np.mean(np.square(Gen.scale_c * (y_mid - y_true))))

    print(f'  PICP={PICP:.4f}  MPIW={MPIW:.4f}  RMSE_piven={RMSE:.4f}  RMSE_mid={RMSE_mid:.4f}')

    results_runs.append((PICP, MPIW, RMSE, RMSE_mid))

results = np.array(results_runs)
avg = np.mean(results, axis=0)
std = np.std(results, axis=0, ddof=1 if n_runs > 1 else 0)

# Save results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results-sp-piven')
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(
    results_dir,
    f"{args.dataset}-{start_time.strftime('%d-%m-%H-%M')}-piven-baseline.csv")

col_names = ['PICP', 'MPIW', 'RMSE_piven', 'RMSE_mid']
rows = {'metric': col_names,
        'avg': [round(a, 4) for a in avg],
        'std': [round(s, 4) for s in std]}
df = pd.DataFrame(rows)
df.to_csv(results_path, index=False)

# Print summary
print(f'\n\n{"="*70}')
print(f'PIVEN BASELINE SUMMARY — {args.dataset} — {n_runs} runs, no ensemble')
print(f'{"="*70}')
print(f'  PICP:       {avg[0]:.4f} ± {std[0]:.4f}')
print(f'  MPIW:       {avg[1]:.4f} ± {std[1]:.4f}')
print(f'  RMSE_piven: {avg[2]:.4f} ± {std[2]:.4f}')
print(f'  RMSE_mid:   {avg[3]:.4f} ± {std[3]:.4f}')

end_time = datetime.datetime.now()
total_time = end_time - start_time
print(f'\nMinutes taken: {round(total_time.total_seconds() / 60, 3)}')

with open(results_path, 'a') as f:
    f.write(f'\nminutes_taken,{round(total_time.total_seconds() / 60, 3)}\n')
    f.write(f'dataset,{args.dataset}\n')
    f.write(f'n_runs,{n_runs}\n')
    f.write(f'n_ensemble,{n_ensemble}\n')
    for k, v in params.items():
        f.write(f'{k},{v}\n')
