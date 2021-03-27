# coding=utf-8
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import tensorflow as tf


class ALTrainer(object):
    """
    Augmented Lagrangian method with first-order gradient-based optimization.
    """
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

    def train(self, model, X, graph_thres, max_iter, iter_step):
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
        model.sess.run(tf.global_variables_initializer())
        rho, alpha, h, h_new = self.init_rho, 0.0, np.inf, np.inf
        prev_W_est, prev_mse = None, float('inf')

        for i in range(1, max_iter + 1):
            while rho < self.rho_thres:
                loss_new, mse_new, h_new, W_new = self.train_step(model, iter_step, X, rho, alpha)
                if h_new > self.h_thres * h:
                    rho *= self.rho_multiply
                else:
                    break

            if self.early_stopping:
                if mse_new / prev_mse > self.early_stopping_thres and h_new <= 1e-7:
                    # MSE increase by too much, revert back to original graph and early stopping
                    # Only perform this early stopping when h_new is sufficiently small
                    # (at least smaller than 1e-7)
                    return prev_W_est
                else:
                    prev_W_est = W_new
                    prev_mse = mse_new

            # Evaluate the learned W in each iteration after thresholding
            W_thresholded = np.copy(W_new)
            W_thresholded = W_thresholded / np.max(np.abs(W_thresholded))
            W_thresholded[np.abs(W_thresholded) < graph_thres] = 0

            W_est, h = W_new, h_new
            alpha += rho * h

            if h <= self.h_tol and i > self.init_iter:
                break

        return W_est

    def train_step(self, model, iter_step, X, rho, alpha):
        for _ in range(iter_step):
            _, curr_loss, curr_mse, curr_h, curr_W \
                = model.sess.run([model.train_op, model.loss, model.mse_loss, model.h, model.W_prime],
                                 feed_dict={model.X: X,
                                            model.rho: rho,
                                            model.alpha: alpha,
                                            model.lr: self.learning_rate})

        return curr_loss, curr_mse, curr_h, curr_W
