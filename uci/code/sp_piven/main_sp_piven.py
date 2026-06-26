# -*- coding: utf-8 -*-
"""
Run SP-PIVEN regression experiment.
Run as:
    python main_sp_piven.py --dataset concrete
    python main_sp_piven.py --dataset concrete --coverage 0.8
    python main_sp_piven.py --dataset concrete --aux_mode mse
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
os.chdir(parent_dir)
from DataGen import DataGenerator
from SPPiven import SPPivenNetwork
from sklearn.model_selection import train_test_split

start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='concrete',
                    help='dataset name, from UCI_Datasets folder')
parser.add_argument('--coverage', type=float, default=None,
                    help='run a single target coverage (e.g. 0.8)')
parser.add_argument('--aux_mode', type=str, default=None,
                    help='override aux_mode: piven or mse')
args = parser.parse_args()

# Load params
params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params_sp_piven.json')
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
c_pi = params['c_pi']
alpha = params['alpha']
beta = params['beta']
beta_h = params['beta_h']
lambda_pi = params['lambda_pi']
lambda_sel = params['lambda_sel']
soften = params['soften']
warmup_frac = params['warmup_frac']
aux_mode = args.aux_mode if args.aux_mode else params['aux_mode']
l_rate = params['lr']
decay_rate = params['decay_rate']
sigma_in = params['sigma_in']
weight_decay = params['weight_decay']
out_biases = params['out_biases']
n_batch = params['n_batch']
patience = params['patience']
is_early_stop = patience != -1
train_prop = 0.9

if args.coverage is not None:
    target_coverages = [args.coverage]
else:
    target_coverages = params['target_coverages']


def run_experiment(target_coverage):
    """Run SP-PIVEN for a single target coverage across all runs."""
    print(f'\n{"="*70}')
    print(f'Dataset: {args.dataset} | Target coverage: {target_coverage} | aux_mode: {aux_mode}')
    print(f'{"="*70}')

    results_runs = []

    for run in range(n_runs):
        print(f'\n--- Run {run + 1}/{n_runs} ---')

        Gen = DataGenerator(dataset_name=args.dataset)
        X_train, y_train, X_test, y_test = Gen.create_data(
            seed_in=run, train_prop=train_prop)

        # Split off validation set for calibration
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=run)

        tf.reset_default_graph()
        sess = tf.Session()

        NN = SPPivenNetwork(
            x_size=X_train.shape[1],
            h_size=h_size,
            c_pi=c_pi,
            soften=soften,
            lambda_pi=lambda_pi,
            lambda_sel=lambda_sel,
            target_coverage=target_coverage,
            alpha=alpha,
            beta=beta,
            beta_h=beta_h,
            sigma_in=sigma_in,
            g_hidden_size=g_hidden_size,
            weight_decay=weight_decay,
            out_biases=out_biases,
            aux_mode=aux_mode,
            patience=patience,
            dataset=args.dataset)

        NN.train(sess, X_train, y_train, X_val, y_val,
                 n_epoch=n_epoch,
                 l_rate=l_rate,
                 decay_rate=decay_rate,
                 lambda_pi=lambda_pi,
                 warmup_frac=warmup_frac,
                 is_early_stop=is_early_stop,
                 n_batch=n_batch)

        # Calibrate threshold
        tau = NN.calibrate_threshold(sess, X_val, y_val, target_coverage)

        # Predict on test set
        if aux_mode == 'piven':
            y_loss, U_pred, L_pred, v_pred, g_pred, U_h_pred, L_h_pred, v_h_pred = \
                NN.predict(sess, X_test, y_test)
        else:
            y_loss, U_pred, L_pred, v_pred, g_pred, h_pred = \
                NN.predict(sess, X_test, y_test)

        U_vals = U_pred[:, 0]
        L_vals = L_pred[:, 0]
        v_vals = v_pred[:, 0]
        g_vals = g_pred[:, 0]
        y_true = y_test[:, 0]

        # Apply calibrated threshold
        selected = g_vals >= tau
        n_selected = np.sum(selected)
        realized_coverage = n_selected / len(y_true)

        # PIVEN point prediction
        y_piven = v_vals * U_vals + (1 - v_vals) * L_vals

        # --- Metrics on SELECTED samples ---
        if n_selected > 0:
            # PI metrics
            cap_sel = (U_vals[selected] > y_true[selected]) & (L_vals[selected] < y_true[selected])
            PICP_sel = np.mean(cap_sel)
            MPIW_sel = np.mean(U_vals[selected] - L_vals[selected])
            # Point prediction
            RMSE_sel = np.sqrt(np.mean(np.square(
                Gen.scale_c * (y_piven[selected] - y_true[selected]))))
        else:
            PICP_sel = MPIW_sel = RMSE_sel = float('nan')

        # --- Metrics on ALL samples ---
        cap_all = (U_vals > y_true) & (L_vals < y_true)
        PICP_all = np.mean(cap_all)
        MPIW_all = np.mean(U_vals - L_vals)
        RMSE_all = np.sqrt(np.mean(np.square(
            Gen.scale_c * (y_piven - y_true))))

        # --- Auxiliary head metrics ---
        if aux_mode == 'piven':
            U_h_vals = U_h_pred[:, 0]
            L_h_vals = L_h_pred[:, 0]
            v_h_vals = v_h_pred[:, 0]
            y_piven_h = v_h_vals * U_h_vals + (1 - v_h_vals) * L_h_vals
            cap_h = (U_h_vals > y_true) & (L_h_vals < y_true)
            aux_PICP = np.mean(cap_h)
            aux_MPIW = np.mean(U_h_vals - L_h_vals)
            aux_RMSE = np.sqrt(np.mean(np.square(
                Gen.scale_c * (y_piven_h - y_true))))
        else:
            h_vals = h_pred[:, 0]
            aux_PICP = float('nan')
            aux_MPIW = float('nan')
            aux_RMSE = np.sqrt(np.mean(np.square(
                Gen.scale_c * (h_vals - y_true))))

        print(f'\n  tau={tau:.4f}  coverage={realized_coverage:.3f} (target={target_coverage})')
        print(f'  PICP_sel={PICP_sel:.4f}  MPIW_sel={MPIW_sel:.4f}  RMSE_sel={RMSE_sel:.4f}')
        print(f'  PICP_all={PICP_all:.4f}  MPIW_all={MPIW_all:.4f}  RMSE_all={RMSE_all:.4f}')
        print(f'  aux_PICP={aux_PICP:.4f}  aux_MPIW={aux_MPIW:.4f}  aux_RMSE={aux_RMSE:.4f}')

        results_runs.append((
            realized_coverage, PICP_sel, MPIW_sel, RMSE_sel,
            PICP_all, MPIW_all, RMSE_all,
            aux_PICP, aux_MPIW, aux_RMSE, tau))
        sess.close()

    return np.array(results_runs)


# Run experiments
all_results = {}
for cov in target_coverages:
    results = run_experiment(cov)
    all_results[cov] = results

# Save results
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results-sp-piven')
os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(
    results_dir,
    f"{args.dataset}-{start_time.strftime('%d-%m-%H-%M')}-sp-piven-{aux_mode}.csv")

col_names = ['coverage', 'PICP_sel', 'MPIW_sel', 'RMSE_sel',
             'PICP_all', 'MPIW_all', 'RMSE_all',
             'aux_PICP', 'aux_MPIW', 'aux_RMSE', 'tau']

rows = []
for cov, results in all_results.items():
    avg = np.nanmean(results, axis=0)
    std = np.nanstd(results, axis=0, ddof=1 if n_runs > 1 else 0)
    row = {'target_coverage': cov}
    for i, name in enumerate(col_names):
        row[f'{name}_avg'] = round(avg[i], 4)
        row[f'{name}_std'] = round(std[i], 4)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(results_path, index=False)

# Print summary
print(f'\n\n{"="*90}')
print(f'SUMMARY — {args.dataset} — SP-PIVEN (aux_mode={aux_mode})')
print(f'{"="*90}')
print(f'{"c_sel":>6} {"cov":>6} {"PICP_s":>7} {"MPIW_s":>7} {"RMSE_s":>7} '
      f'{"PICP_a":>7} {"MPIW_a":>7} {"RMSE_a":>7} '
      f'{"auxPICP":>7} {"auxRMSE":>7}')
print('-' * 80)
for row in rows:
    print(f'{row["target_coverage"]:>6.2f} '
          f'{row["coverage_avg"]:>6.3f} '
          f'{row["PICP_sel_avg"]:>7.4f} '
          f'{row["MPIW_sel_avg"]:>7.4f} '
          f'{row["RMSE_sel_avg"]:>7.4f} '
          f'{row["PICP_all_avg"]:>7.4f} '
          f'{row["MPIW_all_avg"]:>7.4f} '
          f'{row["RMSE_all_avg"]:>7.4f} '
          f'{row["aux_PICP_avg"]:>7.4f} '
          f'{row["aux_RMSE_avg"]:>7.4f}')

# Timing
end_time = datetime.datetime.now()
total_time = end_time - start_time
print(f'\nMinutes taken: {round(total_time.total_seconds() / 60, 3)}')
print(f'Start: {start_time.strftime("%H:%M:%S")}  End: {end_time.strftime("%H:%M:%S")}')

with open(results_path, 'a') as f:
    f.write(f'\nminutes_taken,{round(total_time.total_seconds() / 60, 3)}\n')
    f.write(f'dataset,{args.dataset}\n')
    f.write(f'aux_mode,{aux_mode}\n')
    for k, v in params.items():
        if k != 'target_coverages':
            f.write(f'{k},{v}\n')
