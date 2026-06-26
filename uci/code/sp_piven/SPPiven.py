"""
SP-PIVEN: Selective Prediction Intervals with Specific Value Prediction.

Joint architecture combining PIVEN and SelectiveNet (Approach A from proposals).

The backbone and PIVEN heads (U, L, v) match the original PIVEN architecture
(DeepNetPI.py) exactly — same layer sizes, same initialization, no batch norm.
The only additions are:
  - g(x): selection head (with its own batch-normed hidden layer, per SelectiveNet)
  - h(x): auxiliary PIVEN triple trained on all samples

This ensures a fair comparison: the only difference between PIVEN and SP-PIVEN
is the selection mechanism.

Loss:
  L = alpha * [beta * L_selPI + (1-beta) * L_selV] + (1-alpha) * L_aux
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

DEVICE = "/cpu:0"


class SPPivenNetwork:
    def __init__(self, x_size, h_size,
                 c_pi=0.95,
                 soften=160.,
                 lambda_pi=15.,
                 lambda_sel=32.,
                 target_coverage=0.8,
                 alpha=0.5,
                 beta=0.5,
                 beta_h=0.5,
                 sigma_in=0.2,
                 g_hidden_size=16,
                 weight_decay=0.,
                 out_biases=[3., -3.],
                 aux_mode='piven',
                 **kwargs):
        """
        @param x_size: number of input features
        @param h_size: list of hidden layer sizes for the main body (e.g. [50])
        @param c_pi: PI coverage target (e.g. 0.95 = 95% coverage)
        @param soften: sigmoid softening factor for k_soft
        @param lambda_pi: PI coverage penalty weight
        @param lambda_sel: selection coverage penalty weight
        @param target_coverage: desired fraction of samples to accept (c_sel)
        @param alpha: balance between selective loss and auxiliary loss
        @param beta: balance between PI loss and value loss (within selective)
        @param beta_h: balance between PI loss and value loss (within auxiliary)
        @param sigma_in: stddev for weight initialization
        @param g_hidden_size: hidden layer size for selection head
        @param weight_decay: L2 regularization (0 by default, matching PIVEN)
        @param out_biases: initial biases for PI bounds [upper_bias, lower_bias]
        @param aux_mode: 'piven' (full triple) or 'mse' (single head)
        """
        self.aux_mode = aux_mode
        self.patience = kwargs.get('patience', -1)
        self.dataset = kwargs.get('dataset', 'dataset placeholder')

        # --- Placeholders ---
        X = tf.placeholder(tf.float32, [None, x_size], name='X')
        y_true = tf.placeholder(tf.float32, [None, 1], name='y_true')
        is_training = tf.placeholder(tf.bool, name='is_training')
        lambda_pi_ph = tf.placeholder(tf.float32, [], name='lambda_pi')

        with tf.device(DEVICE):
            # === Shared backbone (matches PIVEN's DeepNetPI.py exactly) ===
            # Manual weight construction — no batch norm, ReLU activation
            W = []
            b = []
            layer_in = []
            layer = []

            # First hidden layer
            W.append(tf.Variable(tf.random_normal([x_size, h_size[0]], stddev=sigma_in)))
            b.append(tf.Variable(np.zeros(h_size[0]) + 0.1, dtype=tf.float32))

            # Additional hidden layers (if any)
            for i in range(1, len(h_size)):
                W.append(tf.Variable(tf.random_normal([h_size[i - 1], h_size[i]], stddev=sigma_in)))
                b.append(tf.Variable(np.zeros(h_size[i]) + 0.1, dtype=tf.float32))

            # Build forward pass
            layer_in.append(tf.matmul(X, W[0]) + b[0])
            layer.append(tf.nn.relu(layer_in[-1]))
            for i in range(1, len(h_size)):
                layer_in.append(tf.matmul(layer[i - 1], W[i]) + b[i])
                layer.append(tf.nn.relu(layer_in[-1]))

            body_out = layer[-1]  # output of shared backbone

            # === Primary PIVEN heads (U, L, v) — matches DeepNetPI.py ===
            # U and L: single 2-neuron output layer (shared weight matrix)
            W_pi = tf.Variable(tf.random_normal([h_size[-1], 2], stddev=sigma_in))
            b_pi = tf.Variable(out_biases)
            pi_out = tf.matmul(body_out, W_pi) + b_pi

            # v head: separate sigmoid neuron
            W_v = tf.Variable(tf.random_normal([h_size[-1], 1], stddev=sigma_in))
            b_v = tf.Variable(tf.zeros(1) + 0.01)
            v_out = tf.nn.sigmoid(tf.matmul(body_out, W_v) + b_v)

            y_U = pi_out[:, 0]
            y_L = pi_out[:, 1]
            y_v = v_out[:, 0]

            # === Selection head g (the SelectiveNet addition) ===
            # This head has its own hidden layer with batch norm (per SelectiveNet paper)
            g_hidden = tf.layers.dense(
                body_out, g_hidden_size,
                kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                name='g_dense')
            g_hidden = tf.layers.batch_normalization(
                g_hidden, training=is_training, name='g_bn')
            g_hidden = tf.nn.relu(g_hidden)
            g_out = tf.layers.dense(
                g_hidden, 1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                name='g_head')
            g_val = g_out[:, 0]

            # === Auxiliary head h ===
            if aux_mode == 'piven':
                # Full PIVEN triple on all samples (same structure as primary heads)
                W_pi_h = tf.Variable(tf.random_normal([h_size[-1], 2], stddev=sigma_in))
                b_pi_h = tf.Variable(list(out_biases))
                pi_h_out = tf.matmul(body_out, W_pi_h) + b_pi_h

                W_v_h = tf.Variable(tf.random_normal([h_size[-1], 1], stddev=sigma_in))
                b_v_h = tf.Variable(tf.zeros(1) + 0.01)
                v_h_out = tf.nn.sigmoid(tf.matmul(body_out, W_v_h) + b_v_h)

                y_U_h = pi_h_out[:, 0]
                y_L_h = pi_h_out[:, 1]
                y_v_h = v_h_out[:, 0]
            else:
                W_h = tf.Variable(tf.random_normal([h_size[-1], 1], stddev=sigma_in))
                b_h = tf.Variable(tf.zeros(1))
                h_out_val = tf.matmul(body_out, W_h) + b_h
                h_pred = h_out_val[:, 0]

            # === Loss computation ===
            y_T = y_true[:, 0]
            N_ = tf.cast(tf.size(y_T), tf.float32)

            # Capture indicators (from PIVEN)
            k_soft = tf.multiply(
                tf.sigmoid((y_U - y_T) * soften),
                tf.sigmoid((y_T - y_L) * soften))
            k_hard = tf.multiply(
                tf.maximum(0., tf.sign(y_U - y_T)),
                tf.maximum(0., tf.sign(y_T - y_L)))

            # Key quantities
            M_g_soft = tf.reduce_sum(g_val * k_soft)
            M_g_hard = tf.reduce_sum(g_val * k_hard)
            S_g = tf.reduce_sum(g_val)
            phi_g = S_g / N_

            # --- Selective PI loss (L_selPI) ---
            MPIW_capt_sel = tf.reduce_sum(
                tf.abs(y_U - y_L) * k_hard * g_val) / (M_g_hard + 1e-6)

            PICP_sel_soft = M_g_soft / (S_g + 1e-6)

            pi_penalty = tf.sqrt(S_g + 1e-6) * lambda_pi_ph * tf.square(
                tf.maximum(0., c_pi - PICP_sel_soft))

            sel_penalty = lambda_sel * tf.square(
                tf.maximum(0., target_coverage - phi_g))

            L_selPI = MPIW_capt_sel + pi_penalty + sel_penalty

            # --- Selective value loss (L_selV) ---
            y_hat = y_v * y_U + (1. - y_v) * y_L
            sample_mse = tf.square(y_hat - y_T)
            L_selV = tf.reduce_sum(g_val * sample_mse) / (S_g + 1e-6)

            # --- Auxiliary loss (L_aux) ---
            if aux_mode == 'piven':
                k_soft_h = tf.multiply(
                    tf.sigmoid((y_U_h - y_T) * soften),
                    tf.sigmoid((y_T - y_L_h) * soften))
                k_hard_h = tf.multiply(
                    tf.maximum(0., tf.sign(y_U_h - y_T)),
                    tf.maximum(0., tf.sign(y_T - y_L_h)))

                MPIW_h = tf.reduce_sum(
                    tf.abs(y_U_h - y_L_h) * k_hard_h) / (tf.reduce_sum(k_hard_h) + 1e-6)
                PICP_h_soft = tf.reduce_mean(k_soft_h)
                L_PI_h = MPIW_h + lambda_pi_ph * tf.sqrt(N_) * tf.square(
                    tf.maximum(0., c_pi - PICP_h_soft))

                y_hat_h = y_v_h * y_U_h + (1. - y_v_h) * y_L_h
                L_v_h = tf.reduce_mean(tf.square(y_hat_h - y_T))

                L_aux = beta_h * L_PI_h + (1. - beta_h) * L_v_h
            else:
                L_aux = tf.reduce_mean(tf.square(h_pred - y_T))

            # --- L2 regularization (optional, off by default to match PIVEN) ---
            if weight_decay > 0:
                l2_vars = tf.trainable_variables()
                l2_loss = weight_decay * tf.add_n(
                    [tf.nn.l2_loss(v) for v in l2_vars if 'bias' not in v.name])
            else:
                l2_loss = 0.

            # --- Final objective ---
            L_selective = beta * L_selPI + (1. - beta) * L_selV
            loss = alpha * L_selective + (1. - alpha) * L_aux + l2_loss

        # === Metrics ===
        metric = []
        metric_name = []

        with tf.device(DEVICE):
            hard_sel = tf.cast(g_val >= 0.5, tf.float32)
            hard_coverage = tf.reduce_mean(hard_sel)
            n_hard_sel = tf.reduce_sum(hard_sel) + 1e-6

            PICP_sel_hard = tf.reduce_sum(k_hard * hard_sel) / n_hard_sel
            metric.append(PICP_sel_hard)
            metric_name.append('PICP_sel')

            MPIW_sel_hard = tf.reduce_sum(tf.abs(y_U - y_L) * hard_sel) / n_hard_sel
            metric.append(MPIW_sel_hard)
            metric_name.append('MPIW_sel')

            metric.append(hard_coverage)
            metric_name.append('coverage')

            PICP_all = tf.reduce_mean(k_hard)
            metric.append(PICP_all)
            metric_name.append('PICP_all')

            MPIW_all = tf.reduce_mean(y_U - y_L)
            metric.append(MPIW_all)
            metric_name.append('MPIW_all')

            sel_rmse = tf.sqrt(
                tf.reduce_sum(sample_mse * hard_sel) / n_hard_sel)
            metric.append(sel_rmse)
            metric_name.append('sel_RMSE')

            metric.append(phi_g)
            metric_name.append('soft_cov')

        # === Save references ===
        self.X = X
        self.y_true = y_true
        self.is_training = is_training
        self.lambda_pi_ph = lambda_pi_ph
        self.loss = loss
        self.L_selPI = L_selPI
        self.L_selV = L_selV
        self.L_aux = L_aux
        self.U_out = tf.reshape(y_U, [-1, 1])
        self.L_out = tf.reshape(y_L, [-1, 1])
        self.v_out = tf.reshape(y_v, [-1, 1])
        self.g_out = g_out
        if aux_mode == 'piven':
            self.U_h_out = tf.reshape(y_U_h, [-1, 1])
            self.L_h_out = tf.reshape(y_L_h, [-1, 1])
            self.v_h_out = tf.reshape(y_v_h, [-1, 1])
        else:
            self.h_out = h_out_val
        self.metric = metric
        self.metric_name = metric_name

    def train(self, sess, X_train, y_train, X_val, y_val,
              n_epoch, l_rate=0.03, n_batch=100,
              decay_rate=0.98, lambda_pi=15.,
              warmup_frac=0.2,
              is_early_stop=False,
              is_print_info=True):
        """Train the SP-PIVEN model with warm-start for lambda_pi."""

        warmup_epoch = int(warmup_frac * n_epoch)
        self.lambda_pi_target = lambda_pi

        global_step = tf.Variable(0, trainable=False)
        decayed_l_rate = tf.train.exponential_decay(
            l_rate, global_step,
            decay_steps=50, decay_rate=decay_rate, staircase=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.device(DEVICE):
                optimizer = tf.train.AdamOptimizer(learning_rate=decayed_l_rate)
                train_step = optimizer.minimize(self.loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        loss_log = []
        val_loss_prev = float('inf')
        patience_counter = 0

        for epoch in range(n_epoch):
            current_lambda_pi = 0.0 if epoch < warmup_epoch else lambda_pi

            if epoch == warmup_epoch and is_print_info:
                print(f'\n  === WARM-START OVER (epoch {epoch}): '
                      f'enabling PI coverage penalty (lambda_pi={lambda_pi}) ===')

            perm = np.random.permutation(X_train.shape[0])
            X_shuff = X_train[perm]
            y_shuff = y_train[perm]

            loss_train = 0
            n_batches = max(1, int(round(X_train.shape[0] / n_batch)))
            for b_idx in range(n_batches):
                if b_idx == n_batches - 1:
                    X_b = X_shuff[b_idx * n_batch:]
                    y_b = y_shuff[b_idx * n_batch:]
                else:
                    X_b = X_shuff[b_idx * n_batch:(b_idx + 1) * n_batch]
                    y_b = y_shuff[b_idx * n_batch:(b_idx + 1) * n_batch]

                _, loss_b = sess.run(
                    [train_step, self.loss],
                    feed_dict={self.X: X_b, self.y_true: y_b,
                               self.is_training: True,
                               self.lambda_pi_ph: current_lambda_pi})
                loss_train += loss_b / n_batches

            if epoch % max(1, int(n_epoch / 10)) == 0 or epoch == n_epoch - 1:
                feed_val = {self.X: X_val, self.y_true: y_val,
                            self.is_training: False,
                            self.lambda_pi_ph: current_lambda_pi}

                loss_val, l_pi, l_v, l_a = sess.run(
                    [self.loss, self.L_selPI, self.L_selV, self.L_aux],
                    feed_dict=feed_val)
                l_rate_epoch = sess.run(decayed_l_rate)

                if is_print_info:
                    print(f'\nep: {epoch}  \ttrn {loss_train:.4f}  '
                          f'val {loss_val:.4f}  '
                          f'[PI {l_pi:.4f} V {l_v:.4f} aux {l_a:.4f}]',
                          end='\t')

                    metric_vals = sess.run(self.metric, feed_dict=feed_val)
                    for name, val in zip(self.metric_name, metric_vals):
                        print(f'{name} {val:.4f}', end='  ')
                    print(f'lr {l_rate_epoch:.6f}', end='')

                loss_log.append((epoch, loss_train, loss_val))

            if is_early_stop:
                val_loss = sess.run(
                    self.loss,
                    feed_dict={self.X: X_val, self.y_true: y_val,
                               self.is_training: False,
                               self.lambda_pi_ph: current_lambda_pi})
                if val_loss >= val_loss_prev:
                    patience_counter += 1
                else:
                    patience_counter = 0
                if patience_counter > self.patience:
                    if is_print_info:
                        print(f'\n\t\t========== EARLY STOP AT EPOCH {epoch} ==========\n')
                    break
                val_loss_prev = val_loss

        self.loss_log = np.array(loss_log)

    def predict(self, sess, X_test, y_test):
        """Run prediction. Returns all head outputs."""
        feed = {self.X: X_test, self.y_true: y_test,
                self.is_training: False,
                self.lambda_pi_ph: self.lambda_pi_target}

        U_pred, L_pred, v_pred, g_pred = sess.run(
            [self.U_out, self.L_out, self.v_out, self.g_out],
            feed_dict=feed)
        y_loss = sess.run(self.loss, feed_dict=feed)

        if self.aux_mode == 'piven':
            U_h_pred, L_h_pred, v_h_pred = sess.run(
                [self.U_h_out, self.L_h_out, self.v_h_out],
                feed_dict=feed)
            return y_loss, U_pred, L_pred, v_pred, g_pred, U_h_pred, L_h_pred, v_h_pred
        else:
            h_pred = sess.run(self.h_out, feed_dict=feed)
            return y_loss, U_pred, L_pred, v_pred, g_pred, h_pred

    def calibrate_threshold(self, sess, X_val, y_val, target_coverage):
        """
        Post-training calibration: find threshold tau such that
        the fraction of samples with g(x) >= tau equals target_coverage.
        """
        g_pred = sess.run(
            self.g_out,
            feed_dict={self.X: X_val, self.y_true: y_val,
                       self.is_training: False,
                       self.lambda_pi_ph: self.lambda_pi_target})
        g_vals = g_pred[:, 0]
        tau = np.percentile(g_vals, 100 * (1 - target_coverage))
        return tau
