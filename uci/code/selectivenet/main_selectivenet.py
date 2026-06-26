# -*- coding: utf-8 -*-
"""
Run SelectiveNet regression experiment.
Run as:
    python main_selectivenet.py --dataset concrete
    python main_selectivenet.py --dataset concrete --coverage 0.8
"""
import argparse
import json
import datetime
import os
import sys
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import pandas as pd

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(parent_dir)
# DataGen uses a relative DATA_PATH — run from the parent dir so it resolves
os.chdir(parent_dir)
from DataGen import DataGenerator
from SelectiveNet import SelectiveNetwork
from sklearn.model_selection import train_test_split

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='concrete',
                    help='dataset name, from UCI_Datasets folder')
parser.add_argument('--coverage', type=float, default=None,
                    help='run a single target coverage (e.g. 0.8). If omitted, runs all coverages.')
args = parser.parse_args()

# load params
params_path = os.path.join(os.path.dirname(__file__), 'params_selectivenet.json')
with open(params_path) as f:
    all_params = json.load(f)
    try:
        params = next(el for el in all_params if el['dataset'] == args.dataset)
    except StopIteration:
        raise ValueError(f"Invalid dataset name: {args.dataset}")

n_runs = params['n_runs']
n_epoch = params['epochs']
h_size = params['h_size']
g_hidden_size = params['g_hidden_size']
lambda_sel = params['lambda_sel']
alpha = params['alpha']
l_rate = params['lr']
decay_rate = params['decay_rate']
sigma_in = params['sigma_in']
weight_decay = params['weight_decay']
n_batch = params['n_batch']
patience = params['patience']
is_early_stop = patience != -1
train_prop = 0.9

if args.coverage is not None:
    target_coverages = [args.coverage]
else:
    target_coverages = params['target_coverages']


def run_experiment(target_coverage):
    """Run SelectiveNet for a single target coverage across all runs."""
    print(f'\n{"="*60}')
    print(f'Dataset: {args.dataset} | Target coverage: {target_coverage}')
    print(f'{"="*60}')

    results_runs = []

    for run in range(n_runs):
        print(f'\n--- Run {run + 1}/{n_runs} ---')

        Gen = DataGenerator(dataset_name=args.dataset)
        X_train, y_train, X_test, y_test = Gen.create_data(
            seed_in=run, train_prop=train_prop)

        # Split off a validation set for calibration
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=run)

        tf.reset_default_graph()
        sess = tf.Session()

        NN = SelectiveNetwork(
            x_size=X_train.shape[1],
            h_size=h_size,
            target_coverage=target_coverage,
            lambda_sel=lambda_sel,
            alpha=alpha,
            sigma_in=sigma_in,
            g_hidden_size=g_hidden_size,
            weight_decay=weight_decay,
            patience=patience,
            dataset=args.dataset)

        NN.train(sess, X_train, y_train, X_val, y_val,
                 n_epoch=n_epoch,
                 l_rate=l_rate,
                 decay_rate=decay_rate,
                 is_early_stop=is_early_stop,
                 n_batch=n_batch)

        # Calibrate threshold on validation set
        tau = NN.calibrate_threshold(sess, X_val, y_val, target_coverage)

        # Predict on test set
        y_loss, f_pred, g_pred, h_pred = NN.predict(sess, X_test, y_test)

        f_vals = f_pred[:, 0]
        g_vals = g_pred[:, 0]
        y_true = y_test[:, 0]

        # Apply calibrated threshold
        selected = g_vals >= tau
        n_selected = np.sum(selected)
        realized_coverage = n_selected / len(y_true)

        # Selective MSE/RMSE (only on selected samples)
        if n_selected > 0:
            sel_mse = np.mean(np.square(Gen.scale_c * (f_vals[selected] - y_true[selected])))
            sel_rmse = np.sqrt(sel_mse)
        else:
            sel_rmse = float('nan')

        # Full RMSE (on all samples, for reference)
        full_mse = np.mean(np.square(Gen.scale_c * (f_vals - y_true)))
        full_rmse = np.sqrt(full_mse)

        # Auxiliary head RMSE
        h_vals = h_pred[:, 0]
        aux_mse = np.mean(np.square(Gen.scale_c * (h_vals - y_true)))
        aux_rmse = np.sqrt(aux_mse)

        print(f'\n  tau={tau:.4f}  coverage={realized_coverage:.4f} '
              f'(target={target_coverage})  sel_RMSE={sel_rmse:.4f}  '
              f'full_RMSE={full_rmse:.4f}  aux_RMSE={aux_rmse:.4f}')

        results_runs.append((realized_coverage, sel_rmse, full_rmse, aux_rmse, tau))
        sess.close()

    return np.array(results_runs)


# Run experiments and save results
all_results = {}
for cov in target_coverages:
    results = run_experiment(cov)
    all_results[cov] = results

# Save results
results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results-selectivenet')
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(
    results_dir,
    f"{args.dataset}-{start_time.strftime('%d-%m-%H-%M')}-selectivenet.csv")

rows = []
for cov, results in all_results.items():
    avg = np.nanmean(results, axis=0)
    std = np.nanstd(results, axis=0, ddof=1 if n_runs > 1 else 0)
    rows.append({
        'target_coverage': cov,
        'realized_coverage_avg': round(avg[0], 4),
        'realized_coverage_std': round(std[0], 4),
        'sel_RMSE_avg': round(avg[1], 4),
        'sel_RMSE_std': round(std[1], 4),
        'full_RMSE_avg': round(avg[2], 4),
        'full_RMSE_std': round(std[2], 4),
        'aux_RMSE_avg': round(avg[3], 4),
        'aux_RMSE_std': round(std[3], 4),
        'tau_avg': round(avg[4], 4),
        'tau_std': round(std[4], 4),
    })

df = pd.DataFrame(rows)
df.to_csv(results_path, index=False)

# Print summary table
print(f'\n\n{"="*80}')
print(f'SUMMARY — {args.dataset} — SelectiveNet')
print(f'{"="*80}')
print(f'{"Coverage":>10} {"Realized":>10} {"Sel RMSE":>12} {"Full RMSE":>12} {"Aux RMSE":>12}')
print(f'{" (target)":>10} {"Coverage":>10} {"(selected)":>12} {"(all)":>12} {"":>12}')
print('-' * 60)
for row in rows:
    print(f'{row["target_coverage"]:>10.2f} '
          f'{row["realized_coverage_avg"]:>10.4f} '
          f'{row["sel_RMSE_avg"]:>12.4f} '
          f'{row["full_RMSE_avg"]:>12.4f} '
          f'{row["aux_RMSE_avg"]:>12.4f}')

# Timing
end_time = datetime.datetime.now()
total_time = end_time - start_time
print(f'\nMinutes taken: {round(total_time.total_seconds() / 60, 3)}')
print(f'Start: {start_time.strftime("%H:%M:%S")}  End: {end_time.strftime("%H:%M:%S")}')

with open(results_path, 'a') as f:
    f.write(f'\nminutes_taken,{round(total_time.total_seconds() / 60, 3)}\n')
    f.write(f'dataset,{args.dataset}\n')
    for k, v in params.items():
        if k != 'target_coverages':
            f.write(f'{k},{v}\n')
