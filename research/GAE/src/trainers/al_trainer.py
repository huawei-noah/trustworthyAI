import logging
import numpy as np
import tensorflow as tf

from helpers.dir_utils import create_dir
from helpers.analyze_utils import count_accuracy


class ALTrainer(object):
    """
    Augmented Lagrangian method with first-order gradient-based optimization
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, init_rho, rho_thres, h_thres, rho_multiply, init_iter,
                 learning_rate, h_tol, early_stopping, early_stopping_thres):
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.learning_rate = learning_rate
        self.h_tol = h_tol
        self.early_stopping = early_stopping
        self.early_stopping_thres = early_stopping_thres

    def train(self, model, X, W, graph_thres, max_iter, iter_step, output_dir):
        """
        model object should contain the several class member:
        - sess
        - train_op
        - loss
        - mse_loss
        - h
        - W_prime
        - X
        - rho
        - alpha
        - lr
        """
        # Create directory to save the raw recovered graph in each iteration
        create_dir('{}/raw_recovered_graph'.format(output_dir))

        model.sess.run(tf.global_variables_initializer())
        rho, alpha, h, h_new = self.init_rho, 0.0, np.inf, np.inf
        prev_W_est, prev_mse = None, float('inf')

        self._logger.info('Started training for {} iterations'.format(max_iter))
        for i in range(1, max_iter + 1):
            while rho < self.rho_thres:
                self._logger.info('rho {:.3E}, alpha {:.3E}'.format(rho, alpha))
                loss_new, mse_new, h_new, W_new = self.train_step(model, iter_step, X, rho, alpha)
                if h_new > self.h_thres * h:
                    rho *= self.rho_multiply
                else:
                    break

            if self.early_stopping:
                if mse_new / prev_mse > self.early_stopping_thres and h_new <= 1e-7:
                    # MSE increases too much, revert back to original graph and perform early stopping
                    # Only perform this early stopping when h_new is sufficiently small
                    # (i.e., at least smaller than 1e-7)
                    return prev_W_est
                else:
                    prev_W_est = W_new
                    prev_mse = mse_new

            # Intermediate outputs
            self.log_and_save_intermediate_outputs(i, W, W_new, graph_thres, loss_new, mse_new, h_new, output_dir)

            W_est, h = W_new, h_new
            alpha += rho * h
            if h <= self.h_tol and i > self.init_iter:
                self._logger.info('Early stopping at {}-th iteration'.format(i))
                break

        # Save model
        model_dir = '{}/model/'.format(output_dir)
        model.save(model_dir)
        self._logger.info('Model saved to {}'.format(model_dir))

        return W_est

    def train_step(self, model, iter_step, X, rho, alpha):
        """
        Solve the suboptimization problem in each iteration
        """
        for _ in range(iter_step):
            _, curr_loss, curr_mse, curr_h, curr_W \
                = model.sess.run([model.train_op, model.loss, model.mse_loss, model.h, model.W_prime],
                                 feed_dict={model.X: X,
                                            model.rho: rho,
                                            model.alpha: alpha,
                                            model.lr: self.learning_rate})

        return curr_loss, curr_mse, curr_h, curr_W

    def log_and_save_intermediate_outputs(self, i, W_true, W_est, graph_thres, loss, mse, h, output_dir):
        # Evaluate the learned W in each iteration after thresholding
        W_thresholded = np.copy(W_est)
        W_thresholded = W_thresholded / np.max(np.abs(W_thresholded))
        W_thresholded[np.abs(W_thresholded) < graph_thres] = 0
        results = count_accuracy(W_true, W_thresholded)

        # Logging
        self._logger.info('[Iter {}] loss {:.3E}, mse {:.3E}, acyclic {:.3E}, shd {}, tpr {:.3f}, fdr {:.3f}, pred_size {}'.format(
            i, loss, mse, h, results['shd'], results['tpr'], results['fdr'], results['pred_size']))

        # Save the raw recovered graph in each iteration
        np.save('{}/raw_recovered_graph/graph_iteration_{}.npy'.format(output_dir, i), W_est)