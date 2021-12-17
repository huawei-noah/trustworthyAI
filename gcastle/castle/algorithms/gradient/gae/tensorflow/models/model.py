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
from tensorflow.keras.layers import Dense, LeakyReLU

from ..helpers.tf_utils import is_cuda_available


class GAEModel(object):
    def __init__(self, n, d, x_dim, seed=8, num_encoder_layers=1, 
                 num_decoder_layers=1, hidden_size=5, latent_dim=1, 
                 l1_graph_penalty=0, use_float64=False):
        self.n = n
        self.d = d
        self.x_dim = x_dim
        self.seed = seed
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.l1_graph_penalty = l1_graph_penalty
        self.tf_float_type = tf.dtypes.float64 if use_float64 else tf.dtypes.float32        

        # Initializer (for reproducibility)
        self.initializer = tf.keras.initializers.glorot_uniform(seed=self.seed)

        self._build()
        self._init_session()
        self._init_saver()

    def _init_session(self):
        if is_cuda_available():
            # Use GPU
            self.sess = tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.5,
                    allow_growth=True,
                )
            ))
        else:
            self.sess = tf.Session()

    def _init_saver(self):
        self.saver = tf.train.Saver()

    def _build(self):
        tf.reset_default_graph()

        self.rho = tf.placeholder(self.tf_float_type)
        self.alpha = tf.placeholder(self.tf_float_type)
        self.lr = tf.placeholder(self.tf_float_type)

        self.X = tf.placeholder(self.tf_float_type, shape=[self.n, self.d, self.x_dim])
        W = tf.Variable(tf.random.uniform([self.d, self.d], minval=-0.1,maxval=0.1,
                                          dtype=self.tf_float_type, seed=self.seed))

        self.W_prime = self._preprocess_graph(W)
        self.mse_loss = self._get_mse_loss(self.X, self.W_prime)

        self.h = tf.linalg.trace(tf.linalg.expm(self.W_prime * self.W_prime)) - self.d    # Acyclicity
        self.loss = 0.5 / self.n * self.mse_loss \
                    + self.l1_graph_penalty * tf.norm(self.W_prime, ord=1) \
                    + self.alpha * self.h + 0.5 * self.rho * self.h * self.h

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _preprocess_graph(self, W):
        # Mask the diagonal entries of graph
        return tf.matrix_set_diag(W, tf.zeros(W.shape[0], dtype=self.tf_float_type))

    def _get_mse_loss(self, X, W_prime):
        X_prime = self._encoder_forward(X)
        X_prime = tf.einsum('ijk,jl->ilk', X_prime, W_prime)        
        X_prime = self._decoder_forward(X_prime)

        return tf.square(tf.linalg.norm(X - X_prime))

    def _encoder_forward(self, x):
        return self._flatten_forward(x, self.num_encoder_layers, self.hidden_size,
                                     input_dim=self.x_dim,
                                     output_dim=self.latent_dim)

    def _decoder_forward(self, x):
        return self._flatten_forward(x, self.num_decoder_layers, self.hidden_size,
                                     input_dim=self.latent_dim,
                                     output_dim=self.x_dim)

    def _flatten_forward(self, x, num_hidden_layers, hidden_size, input_dim, output_dim):
        if num_hidden_layers == 0:
            return x
        else:
            flatten = tf.reshape(x, shape=(-1, input_dim))
            for _ in range(num_hidden_layers):    # Hidden layer
                flatten = Dense(hidden_size, activation=None, kernel_initializer=self.initializer)(flatten)
                flatten = LeakyReLU(alpha=0.05)(flatten)

            flatten = Dense(output_dim, kernel_initializer=self.initializer)(flatten)    # Final output layer

            return tf.reshape(flatten, shape=(self.n, self.d, output_dim))

