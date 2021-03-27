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

import logging
import numpy as np
import tensorflow as tf

from ..helpers.analyze_utils import compute_acyclicity, convert_logits_to_sigmoid


class ALTrainer(object):
    """
    Augmented Lagrangian method with first-order gradient-based optimization
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, init_rho, rho_thres, h_thres, rho_multiply,
                 init_iter, h_tol, temperature):
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.temperature = temperature

    def train(self, model, X, max_iter, iter_step):
        """
        model object should contain the following class members:
        - sess
        - train_op
        - loss
        - mse_loss
        - h
        - W_prime
        - W
        - X
        - rho
        - alpha
        - temperature
        """
        model.sess.run(tf.compat.v1.global_variables_initializer())
        rho, alpha, h, h_new = self.init_rho, 0.0, np.inf, np.inf

        for i in range(1, max_iter + 1):
            while rho < self.rho_thres:
                loss_new, mse_new, h_new, W_logits_new \
                    = self.train_step(model, iter_step, X, rho, alpha, self.temperature)
                if h_new > self.h_thres * h:
                    rho *= self.rho_multiply
                else:
                    break

            # Use two stopping criterions
            h_logits = compute_acyclicity(convert_logits_to_sigmoid(W_logits_new/self.temperature))
            if h_new <= self.h_tol and h_logits <= self.h_tol and i > self.init_iter:
                break

            # Update h and alpha
            h = h_new
            alpha += rho * h_new

        return W_logits_new

    def train_step(self, model, iter_step, X, rho, alpha, temperature):
        # curr_loss, curr_mse and curr_h are single-sample estimation
        for _ in range(iter_step):
            _, curr_loss, curr_mse, curr_h, curr_W_sampled, curr_W_logits \
                = model.sess.run([model.train_op, model.loss, model.mse_loss, model.h, model.W_prime, model.W],
                                 feed_dict={model.X: X,
                                            model.rho: rho,
                                            model.alpha: alpha,
                                            model.tau: temperature})

        return curr_loss, curr_mse, curr_h, curr_W_logits


