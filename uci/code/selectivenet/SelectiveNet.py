"""
SelectiveNet for regression (TensorFlow 1.x).

Architecture follows the paper:
  - Main body: FC hidden layers with batch norm + ReLU
  - Prediction head f: 1 linear neuron
  - Selection head g: FC hidden layer (16 neurons, batch norm, ReLU) -> 1 sigmoid neuron
  - Auxiliary head h: 1 linear neuron (same task as f, trained on all samples)

Loss:
  L = alpha * L_(f,g) + (1 - alpha) * L_h
  where L_(f,g) = selective_risk + lambda * max(0, c - coverage)^2
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

DEVICE = "/cpu:0"


class SelectiveNetwork:
    def __init__(self, x_size, h_size,
                 target_coverage=0.8,
                 lambda_sel=32.,
                 alpha=0.5,
                 sigma_in=0.1,
                 g_hidden_size=16,
                 weight_decay=1e-4,
                 **kwargs):
        """
        @param x_size: number of input features
        @param h_size: list of hidden layer sizes for the main body
        @param target_coverage: desired fraction of samples to accept (c in the paper)
        @param lambda_sel: penalty weight for coverage constraint
        @param alpha: balance between selective loss and auxiliary loss
        @param sigma_in: stddev for weight initialization
        @param g_hidden_size: hidden layer size for the selection head
        @param weight_decay: L2 regularization coefficient
        """
        self.target_coverage = target_coverage
        self.patience = kwargs.get('patience', -1)
        self.dataset = kwargs.get('dataset', 'dataset placeholder')

        X = tf.placeholder(tf.float32, [None, x_size], name='X')
        y_true = tf.placeholder(tf.float32, [None, 1], name='y_true')
        is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.device(DEVICE):
            # === Main body block ===
            body_out = X
            for i, units in enumerate(h_size):
                body_out = tf.layers.dense(
                    body_out, units,
                    kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                    name=f'body_dense_{i}')
                body_out = tf.layers.batch_normalization(
                    body_out, training=is_training, name=f'body_bn_{i}')
                body_out = tf.nn.relu(body_out)

            # === Prediction head f: 1 linear neuron ===
            f_out = tf.layers.dense(
                body_out, 1,
                kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                name='f_head')

            # === Selection head g: hidden layer -> sigmoid ===
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

            # === Auxiliary head h: 1 linear neuron ===
            h_out = tf.layers.dense(
                body_out, 1,
                kernel_initializer=tf.random_normal_initializer(stddev=sigma_in),
                name='h_head')

            # === Loss computation ===
            y_T = y_true[:, 0]
            f_pred = f_out[:, 0]
            g_val = g_out[:, 0]
            h_pred = h_out[:, 0]

            # Empirical coverage: mean of g(x)
            empirical_coverage = tf.reduce_mean(g_val)

            # Selective risk: weighted MSE where weights are g(x)
            # r_hat_A = sum(loss * g) / sum(g)
            sample_losses = tf.square(f_pred - y_T)
            selective_risk = tf.divide(
                tf.reduce_sum(sample_losses * g_val),
                tf.reduce_sum(g_val) + 1e-6)

            # Coverage penalty
            coverage_penalty = lambda_sel * tf.square(
                tf.maximum(0., target_coverage - empirical_coverage))

            # Selective loss
            L_fg = selective_risk + coverage_penalty

            # Auxiliary loss: standard MSE on all samples
            L_h = tf.reduce_mean(tf.square(h_pred - y_T))

            # L2 regularization
            l2_vars = tf.trainable_variables()
            l2_loss = weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in l2_vars if 'bias' not in v.name])

            # Total loss
            loss = alpha * L_fg + (1. - alpha) * L_h + l2_loss

        # Metrics
        metric = []
        metric_name = []

        with tf.device(DEVICE):
            # Coverage metric (fraction with g >= 0.5)
            hard_selection = tf.cast(g_val >= 0.5, tf.float32)
            hard_coverage = tf.reduce_mean(hard_selection)
            metric.append(hard_coverage)
            metric_name.append('coverage')

            # Selective RMSE (on hard-selected samples)
            selected_losses = sample_losses * hard_selection
            n_selected = tf.reduce_sum(hard_selection) + 1e-6
            sel_mse = tf.reduce_sum(selected_losses) / n_selected
            sel_rmse = tf.sqrt(sel_mse)
            metric.append(sel_rmse)
            metric_name.append('sel_RMSE')

            # Full RMSE (on all samples, for comparison)
            full_rmse = tf.sqrt(tf.reduce_mean(sample_losses))
            metric.append(full_rmse)
            metric_name.append('full_RMSE')

            # Soft coverage (mean of g values)
            metric.append(empirical_coverage)
            metric_name.append('soft_cov')

        # Save references
        self.X = X
        self.y_true = y_true
        self.is_training = is_training
        self.loss = loss
        self.f_out = f_out
        self.g_out = g_out
        self.h_out = h_out
        self.metric = metric
        self.metric_name = metric_name

    def train(self, sess, X_train, y_train, X_val, y_val,
              n_epoch, l_rate=5e-4, n_batch=256,
              decay_rate=0.95, is_early_stop=False,
              is_print_info=True):
        """Train the SelectiveNet model."""

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
            perm = np.random.permutation(X_train.shape[0])
            X_train_shuff = X_train[perm]
            y_train_shuff = y_train[perm]

            loss_train = 0
            n_batches = max(1, int(round(X_train.shape[0] / n_batch)))
            for b in range(n_batches):
                if b == n_batches - 1:
                    X_b = X_train_shuff[b * n_batch:]
                    y_b = y_train_shuff[b * n_batch:]
                else:
                    X_b = X_train_shuff[b * n_batch:(b + 1) * n_batch]
                    y_b = y_train_shuff[b * n_batch:(b + 1) * n_batch]

                _, loss_b = sess.run(
                    [train_step, self.loss],
                    feed_dict={self.X: X_b, self.y_true: y_b, self.is_training: True})
                loss_train += loss_b / n_batches

            # Print info at regular intervals
            if epoch % max(1, int(n_epoch / 10)) == 0 or epoch == n_epoch - 1:
                loss_val = sess.run(
                    self.loss,
                    feed_dict={self.X: X_val, self.y_true: y_val, self.is_training: False})
                l_rate_epoch = sess.run(decayed_l_rate)

                if is_print_info:
                    print(f'\nep: {epoch}  \ttrn loss {loss_train:.4f}  \tval loss {loss_val:.4f}', end='\t')

                    metric_vals = sess.run(
                        self.metric,
                        feed_dict={self.X: X_val, self.y_true: y_val, self.is_training: False})
                    for name, val in zip(self.metric_name, metric_vals):
                        print(f'{name} {val:.4f}', end='\t')
                    print(f'lr {l_rate_epoch:.6f}', end='')

                loss_log.append((epoch, loss_train, loss_val))

            # Early stopping
            if is_early_stop:
                val_loss = sess.run(
                    self.loss,
                    feed_dict={self.X: X_val, self.y_true: y_val, self.is_training: False})
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
        """Run prediction. Returns f(x), g(x), h(x) and loss."""
        f_pred, g_pred, h_pred = sess.run(
            [self.f_out, self.g_out, self.h_out],
            feed_dict={self.X: X_test, self.y_true: y_test, self.is_training: False})
        y_loss = sess.run(
            self.loss,
            feed_dict={self.X: X_test, self.y_true: y_test, self.is_training: False})
        return y_loss, f_pred, g_pred, h_pred

    def calibrate_threshold(self, sess, X_val, y_val, target_coverage):
        """
        Post-training calibration (Section 5 of SelectiveNet paper).
        Find threshold tau such that the fraction of samples with g(x) >= tau
        equals the target coverage.
        """
        g_pred = sess.run(
            self.g_out,
            feed_dict={self.X: X_val, self.y_true: y_val, self.is_training: False})
        g_vals = g_pred[:, 0]
        tau = np.percentile(g_vals, 100 * (1 - target_coverage))
        return tau
