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

from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

from ..helpers.tf_utils import gumbel_sigmoid
from ..helpers.tf_utils import print_summary

class MaskedModel(ABC):
    """
    Seems like there is some randomess in Adam optimizer on GPU, which results in
    non-deterministic behavior even when all seeds are fixed
    
    References:
    - https://stackoverflow.com/questions/53119642/tensorflow-determinism-of-adamoptimizer-when-running-in-cpu
    - https://github.com/keras-team/keras/issues/2280#issuecomment-529966612
    - https://github.com/tensorflow/tensorflow/issues/12871
    - https://github.com/keras-team/keras/issues/439
    - https://github.com/keras-team/keras/issues/12247
    """
    def __init__(self, n, d, pns_mask, num_hidden_layers, hidden_size,
                 l1_graph_penalty, learning_rate, seed, use_float64, use_gpu):
        self.print_summary = print_summary    # Print summary for tensorflow variables

        self.n = n
        self.d = d
        self.pns_mask = pns_mask
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.l1_graph_penalty = l1_graph_penalty
        self.learning_rate = learning_rate
        self.seed = seed
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32
        self.use_gpu = use_gpu

        # Initializer (for reproducibility)
        self.initializer = tf.keras.initializers.glorot_uniform(seed=self.seed)

        self._build()
        self._init_session()
        self._init_saver()

    def _init_session(self):
        if self.use_gpu:
            # Use GPU
            self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                gpu_options=tf.compat.v1.GPUOptions(
                    allow_growth=True,
                )
            ))
        else:
            self.sess = tf.compat.v1.Session()

    def _init_saver(self):
        self.saver = tf.compat.v1.train.Saver()

    def _preprocess_graph(self, W):
        W_prob = gumbel_sigmoid(W, temperature=self.tau, seed=self.seed, tf_float_type=self.tf_float_type)
        W_prob = tf.linalg.set_diag(W_prob, tf.zeros(W_prob.shape[0], dtype=self.tf_float_type))
        return W_prob

    def _build(self):
        tf.compat.v1.reset_default_graph()

        mask = tf.convert_to_tensor(self.pns_mask, dtype=self.tf_float_type)
        self.rho = tf.compat.v1.placeholder(self.tf_float_type)
        self.alpha = tf.compat.v1.placeholder(self.tf_float_type)
        self.tau = tf.compat.v1.placeholder(self.tf_float_type, shape=[])    # Temperature
        self.X = tf.compat.v1.placeholder(self.tf_float_type, shape=[self.n, self.d])
        self.W = tf.compat.v1.Variable(tf.random.uniform([self.d, self.d], minval=-1e-10, maxval=1e-10,
                                                          dtype=self.tf_float_type, seed=self.seed))

        # To be implemented by different models
        self.W_prime = self._preprocess_graph(self.W)
        self.W_prime *= mask    # Preliminary neighborhood selection
        self.mse_loss = self._get_mse_loss(self.X, self.W_prime)

        self.h = tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d
        self.loss = 0.5 / self.n * self.mse_loss \
                    + self.l1_graph_penalty * tf.norm(self.W_prime, ord=1) \
                    + self.alpha * self.h + 0.5 * self.rho * self.h * self.h

        self.train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def _get_mse_loss(self, X, W_prime):
        """
        Different model for different nodes to use masked features to predict value for each node
        """
        mse_loss = 0
        for i in range(self.d):
            # possible_parents = list(range(self.d))
            # possible_parents.remove(i)
            # Get possible PNS parents and also remove diagonal element
            pns_parents = np.where(self.pns_mask[:, i] == 1)[0]
            possible_parents = [int(j) for j in pns_parents if j != i]
            if len(possible_parents) == 0:    # Root node, don't have to build NN in this case
                continue

            curr_X = tf.gather(X, indices=possible_parents, axis=1)    # Features for current node
            curr_y = tf.gather(X, indices=i, axis=1)    # Label for current node
            curr_W = tf.gather(tf.gather(W_prime, indices=i, axis=1),
                               indices=possible_parents, axis=0)    # Mask for current node

            curr_masked_X = curr_X * curr_W    # Broadcasting
            curr_y_pred = self._forward(curr_masked_X)    # Use masked features to predict value of current node

            mse_loss += tf.reduce_sum(tf.square(tf.squeeze(curr_y_pred) - curr_y))

        return mse_loss

    @abstractmethod
    def _forward(self, x):
        pass


