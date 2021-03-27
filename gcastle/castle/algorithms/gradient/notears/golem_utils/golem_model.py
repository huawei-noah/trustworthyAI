# coding=utf-8
# 2021.03 modified (1) logging to loguru
# 2021.03 deleted  (1) __main__
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

from loguru import logger
import tensorflow as tf


class GolemModel:
    """
    Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    def __init__(self, n, d, lambda_1, lambda_2, equal_variances, B_init=None):
        """
        Initialize self.

        Parameters
        ----------
        n: int
            Number of samples.
        d: int
            Number of nodes.
        lambda_1: float
            Coefficient of L1 penalty.
        lambda_2: float
            Coefficient of DAG penalty.
        equal_variances: bool
            Whether to assume equal noise variances
            for likelibood objective. Default: True.
        B_init: numpy.ndarray or None
            [d, d] weighted matrix for initialization. 
            Set to None to disable. Default: None.
        """
        self.n = n
        self.d = d
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init

        self._build()
        self._init_session()

    def _init_session(self):
        """Initialize tensorflow session."""
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True
            )
        ))

    def _build(self):
        """Build tensorflow graph."""
        tf.compat.v1.reset_default_graph()

        # Placeholders and variables
        self.lr = tf.compat.v1.placeholder(tf.float32)
        self.X = tf.compat.v1.placeholder(tf.float32, shape=[self.n, self.d])
        self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        if self.B_init is not None:
            self.B = tf.Variable(tf.convert_to_tensor(self.B_init, tf.float32))
        else:
            self.B = tf.Variable(tf.zeros([self.d, self.d], tf.float32))
        self.B = self._preprocess(self.B)

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h

        # Optimizer
        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.score)
        logger.debug("Finished building tensorflow graph.")

    def _preprocess(self, B):
        """
        Set the diagonals of B to zero.

        Parameters
        ----------
        B: tf.Tensor
            [d, d] weighted matrix.

        Return
        ------
        tf.Tensor: [d, d] weighted matrix.
        """
        return tf.linalg.set_diag(B, tf.zeros(B.shape[0], dtype=tf.float32))

    def _compute_likelihood(self):
        """
        Compute (negative log) likelihood in the linear Gaussian case.

        Return
        ------
        tf.Tensor: Likelihood term (scalar-valued).
        """
        if self.equal_variances:    # Assuming equal noise variances
            return 0.5 * self.d * tf.math.log(
                tf.square(
                    tf.linalg.norm(self.X - self.X @ self.B)
                )
            ) - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]
        else:    # Assuming non-equal noise variances
            return 0.5 * tf.math.reduce_sum(
                tf.math.log(
                    tf.math.reduce_sum(
                        tf.square(self.X - self.X @ self.B), axis=0
                    )
                )
            ) - tf.linalg.slogdet(tf.eye(self.d) - self.B)[1]

    def _compute_L1_penalty(self):
        """
        Compute L1 penalty.

        Return
        ------
        tf.Tensor: L1 penalty term (scalar-valued).
        """
        return tf.norm(self.B, ord=1)

    def _compute_h(self):
        """
        Compute DAG penalty.

        Return
        ------
        tf.Tensor: DAG penalty term (scalar-valued).
        """
        return tf.linalg.trace(tf.linalg.expm(self.B * self.B)) - self.d
