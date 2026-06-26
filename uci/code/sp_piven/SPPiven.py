"""
SP-PIVEN: Selective Prediction Intervals with Specific Value Prediction.

Joint architecture combining PIVEN and SelectiveNet (Approach A from proposals).

Architecture:
  Shared backbone -> 5+ output heads:
    U(x), L(x)  -- PI upper/lower bounds (linear)
    v(x)        -- value interpolator (sigmoid), y_hat = v*U + (1-v)*L
    g(x)        -- selection function (sigmoid), accept if g >= tau
    h(x)        -- auxiliary PIVEN triple (U_h, L_h, v_h) on all samples

Loss:
  L = alpha * [beta * L_selPI + (1-beta) * L_selV] + (1-alpha) * L_aux + L2
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
                 sigma_in=0.1,
                 g_hidden_size=16,
                 weight_decay=1e-4,
                 out_biases=[3., -3.],
                 aux_mode='piven',
                 **kwargs):
        """
        @param x_size: number of input features
        @param h_size: list of hidden layer sizes for the main body
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
        @param weight_decay: L2 regularization coefficient
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
            # === Shared backbone ===
            body_out = X
            for i, units in enumerate(h_size):
                body_out = tf.layers.dense(
                    body_out, units,
                    kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                    name=f'body_dense_{i}')
                body_out = tf.layers.batch_normalization(
                    body_out, training=is_training, name=f'body_bn_{i}')
                body_out = tf.nn.relu(body_out)

            # === Primary PIVEN heads ===
            U_out = tf.layers.dense(
                body_out, 1,
                kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                bias_initializer=tf.constant_initializer(out_biases[0]),
                name='U_head')

            L_out = tf.layers.dense(
                body_out, 1,
                kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                bias_initializer=tf.constant_initializer(out_biases[1]),
                name='L_head')

            v_out = tf.layers.dense(
                body_out, 1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                name='v_head')

            # === Selection head g ===
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

            # === Auxiliary head h ===
            if aux_mode == 'piven':
                U_h_out = tf.layers.dense(
                    body_out, 1,
                    kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                    bias_initializer=tf.constant_initializer(out_biases[0]),
                    name='h_U_head')
                L_h_out = tf.layers.dense(
                    body_out, 1,
                    kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                    bias_initializer=tf.constant_initializer(out_biases[1]),
                    name='h_L_head')
                v_h_out = tf.layers.dense(
                    body_out, 1,
                    activation=tf.nn.sigmoid,
                    kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                    name='h_v_head')
            else:
                h_out = tf.layers.dense(
                    body_out, 1,
                    kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                    name='h_head')

            # === Extract scalars ===
            y_U = U_out[:, 0]
            y_L = L_out[:, 0]
            y_v = v_out[:, 0]
            y_T = y_true[:, 0]
            g_val = g_out[:, 0]
            N_ = tf.cast(tf.size(y_T), tf.float32)

            # === Capture indicators (from PIVEN) ===
            k_soft = tf.multiply(
                tf.sigmoid((y_U - y_T) * soften),
                tf.sigmoid((y_T - y_L) * soften))
            k_hard = tf.multiply(
                tf.maximum(0., tf.sign(y_U - y_T)),
                tf.maximum(0., tf.sign(y_T - y_L)))

            # === Key quantities ===
            # M_g = Σ g(x_i)·k_i  (jointly selected and captured, soft for gradients)
            M_g_soft = tf.reduce_sum(g_val * k_soft)
            # M_g hard (for MPIW)
            M_g_hard = tf.reduce_sum(g_val * k_hard)
            # S_g = Σ g(x_i)  (selected mass)
            S_g = tf.reduce_sum(g_val)
            # φ̂(g) = S_g / n  (empirical selection coverage)
            phi_g = S_g / N_

            # === Selective PI loss (L_selPI) ===
            # MPIW_capt^sel = Σ |U-L|·k_hard·g / M_g_hard
            MPIW_capt_sel = tf.reduce_sum(
                tf.abs(y_U - y_L) * k_hard * g_val) / (M_g_hard + 1e-6)

            # PICP^sel = M_g_soft / S_g (soft version for gradients)
            PICP_sel_soft = M_g_soft / (S_g + 1e-6)

            # PI coverage penalty: √S_g · λ_PI · Ψ(c_PI - PICP^sel)
            pi_penalty = tf.sqrt(S_g + 1e-6) * lambda_pi_ph * tf.square(
                tf.maximum(0., c_pi - PICP_sel_soft))

            # Selection coverage penalty: λ_sel · Ψ(c_sel - φ̂(g))
            sel_penalty = lambda_sel * tf.square(
                tf.maximum(0., target_coverage - phi_g))

            L_selPI = MPIW_capt_sel + pi_penalty + sel_penalty

            # === Selective value loss (L_selV) ===
            y_hat = y_v * y_U + (1. - y_v) * y_L
            sample_mse = tf.square(y_hat - y_T)
            L_selV = tf.reduce_sum(g_val * sample_mse) / (S_g + 1e-6)

            # === Auxiliary loss (L_aux) ===
            if aux_mode == 'piven':
                y_U_h = U_h_out[:, 0]
                y_L_h = L_h_out[:, 0]
                y_v_h = v_h_out[:, 0]

                # Capture indicators for auxiliary head
                k_soft_h = tf.multiply(
                    tf.sigmoid((y_U_h - y_T) * soften),
                    tf.sigmoid((y_T - y_L_h) * soften))
                k_hard_h = tf.multiply(
                    tf.maximum(0., tf.sign(y_U_h - y_T)),
                    tf.maximum(0., tf.sign(y_T - y_L_h)))

                # Standard PIVEN PI loss on all samples
                MPIW_h = tf.reduce_sum(
                    tf.abs(y_U_h - y_L_h) * k_hard_h) / (tf.reduce_sum(k_hard_h) + 1e-6)
                PICP_h_soft = tf.reduce_mean(k_soft_h)
                L_PI_h = MPIW_h + lambda_pi_ph * tf.sqrt(N_) * tf.square(
                    tf.maximum(0., c_pi - PICP_h_soft))

                # Standard PIVEN value loss on all samples
                y_hat_h = y_v_h * y_U_h + (1. - y_v_h) * y_L_h
                L_v_h = tf.reduce_mean(tf.square(y_hat_h - y_T))

                L_aux = beta_h * L_PI_h + (1. - beta_h) * L_v_h
            else:
                h_pred = h_out[:, 0]
                L_aux = tf.reduce_mean(tf.square(h_pred - y_T))

            # === L2 regularization ===
            l2_vars = tf.trainable_variables()
            l2_loss = weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in l2_vars if 'bias' not in v.name])

            # === Final objective ===
            # L = α·[β·L_selPI + (1-β)·L_selV] + (1-α)·L_aux + L2
            L_selective = beta * L_selPI + (1. - beta) * L_selV
            loss = alpha * L_selective + (1. - alpha) * L_aux + l2_loss

        # === Metrics (for monitoring during training) ===
        metric = []
        metric_name = []

        with tf.device(DEVICE):
            # Hard selection (g >= 0.5)
            hard_sel = tf.cast(g_val >= 0.5, tf.float32)
            hard_coverage = tf.reduce_mean(hard_sel)
            n_hard_sel = tf.reduce_sum(hard_sel) + 1e-6

            # PICP on selected (hard)
            PICP_sel_hard = tf.reduce_sum(k_hard * hard_sel) / n_hard_sel
            metric.append(PICP_sel_hard)
            metric_name.append('PICP_sel')

            # MPIW on selected (hard)
            MPIW_sel_hard = tf.reduce_sum(tf.abs(y_U - y_L) * hard_sel) / n_hard_sel
            metric.append(MPIW_sel_hard)
            metric_name.append('MPIW_sel')

            # Coverage
            metric.append(hard_coverage)
            metric_name.append('coverage')

            # PICP on all
            PICP_all = tf.reduce_mean(k_hard)
            metric.append(PICP_all)
            metric_name.append('PICP_all')

            # MPIW on all
            MPIW_all = tf.reduce_mean(y_U - y_L)
            metric.append(MPIW_all)
            metric_name.append('MPIW_all')

            # Selective RMSE (point prediction on selected)
            sel_rmse = tf.sqrt(
                tf.reduce_sum(sample_mse * hard_sel) / n_hard_sel)
            metric.append(sel_rmse)
            metric_name.append('sel_RMSE')

            # Soft coverage
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
        self.U_out = U_out
        self.L_out = L_out
        self.v_out = v_out
        self.g_out = g_out
        if aux_mode == 'piven':
            self.U_h_out = U_h_out
            self.L_h_out = L_h_out
            self.v_h_out = v_h_out
        else:
            self.h_out = h_out
        self.metric = metric
        self.metric_name = metric_name

    def train(self, sess, X_train, y_train, X_val, y_val,
              n_epoch, l_rate=5e-4, n_batch=256,
              decay_rate=0.95, lambda_pi=15.,
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

            # Print warm-start transition
            if epoch == warmup_epoch and is_print_info:
                print(f'\n  === WARM-START OVER (epoch {epoch}): '
                      f'enabling PI coverage penalty (lambda_pi={lambda_pi}) ===')

            # Shuffle and batch
            perm = np.random.permutation(X_train.shape[0])
            X_shuff = X_train[perm]
            y_shuff = y_train[perm]

            loss_train = 0
            n_batches = max(1, int(round(X_train.shape[0] / n_batch)))
            for b in range(n_batches):
                if b == n_batches - 1:
                    X_b = X_shuff[b * n_batch:]
                    y_b = y_shuff[b * n_batch:]
                else:
                    X_b = X_shuff[b * n_batch:(b + 1) * n_batch]
                    y_b = y_shuff[b * n_batch:(b + 1) * n_batch]

                _, loss_b = sess.run(
                    [train_step, self.loss],
                    feed_dict={self.X: X_b, self.y_true: y_b,
                               self.is_training: True,
                               self.lambda_pi_ph: current_lambda_pi})
                loss_train += loss_b / n_batches

            # Print info at intervals
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

            # Early stopping
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
