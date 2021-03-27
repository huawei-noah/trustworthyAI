# coding=utf-8
# 2021.03 modified (1) logging to loguru
# 2021.03 deleted  (1) create_dir
# Huawei Technologies Co., Ltd. 
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Ignavier Ng (https://github.com/ignavier/golem)
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
from loguru import logger


class GolemTrainer:
    """Set up the trainer to solve the unconstrained optimization problem of GOLEM."""

    def __init__(self, learning_rate=1e-3):
        """
        Initialize self.

        Parameters
        ----------
        learning_rate: float
            Learning rate of Adam optimizer. Default: 1e-3.
        """
        self.learning_rate = learning_rate

    def train(self, model, X, num_iter, checkpoint_iter=None):
        """
        Training and checkpointing.

        Parameters
        ----------
        model: GolemModel object
            GolemModel.
        X: numpy.ndarray
            [n, d] data matrix.
        num_iter: int
            Number of iterations for training.
        checkpoint_iter: int
            Number of iterations between each checkpoint.
            Set to None to disable. Default: None.

        Return
        ------
        B_est: numpy.ndarray
            [d, d] estimated weighted matrix.
        """
        model.sess.run(tf.compat.v1.global_variables_initializer())

        logger.info("Started training for {} iterations.".format(int(num_iter)))
        for i in range(0, int(num_iter) + 1):
            if i == 0:    # Do not train here, only perform evaluation
                score, likelihood, h, B_est = self.eval_iter(model, X)
            else:    # Train
                score, likelihood, h, B_est = self.train_iter(model, X)

            if checkpoint_iter is not None and i % checkpoint_iter == 0:
                self.train_checkpoint(i, score, likelihood, h, B_est)

        return B_est

    def eval_iter(self, model, X):
        """
        Evaluation for one iteration. Do not train here.

        Parameters
        ----------
        model: GolemModel object
            GolemModel.
        X: numpy.ndarray
            [n, d] data matrix.

        Return
        ------
        score: float
            value of score function.
        likelihood: float
            value of likelihood function.
        h: float
            value of DAG penalty.
        B_est: numpy.ndarray
            [d, d] estimated weighted matrix.
        """
        score, likelihood, h, B_est = model.sess.run( \
            [model.score, model.likelihood, model.h, model.B], \
            feed_dict={model.X: X, model.lr: self.learning_rate})

        return score, likelihood, h, B_est

    def train_iter(self, model, X):
        """
        Training for one iteration.

        Parameters
        ----------
        model: GolemModel object
            GolemModel.
        X: numpy.ndarray
            [n, d] data matrix.

        Return
        ------
        score: float
            value of score function.
        likelihood: float
            value of likelihood function.
        h: float
            value of DAG penalty.
        B_est: numpy.ndarray
            [d, d] estimated weighted matrix.
        """
        _, score, likelihood, h, B_est = model.sess.run( \
            [model.train_op, model.score, model.likelihood, model.h, model.B], \
            feed_dict={model.X: X, model.lr: self.learning_rate})

        return score, likelihood, h, B_est

    def train_checkpoint(self, i, score, likelihood, h, B_est):
        """
        Log and save intermediate results/outputs.

        Parameters
        ----------
        i: int
            i-th iteration of training.
        score: float
            value of score function.
        likelihood: float
            value of likelihood function.
        h: float
            value of DAG penalty.
        B_est: numpy.ndarray
            [d, d] estimated weighted matrix.
        """
        logger.info("[Iter {}] score={:.3f}, likelihood={:.3f}, h={:.1e}".format( \
            i, score, likelihood, h))
