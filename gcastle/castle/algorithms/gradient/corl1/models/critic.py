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

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
import numpy as np

class Critic(object):
    def __init__(self, config, is_train):
        self.config=config
        self.batch_size = config.batch_size 
        self.max_length = config.max_length

        self.num_neurons = config.hidden_dim 
        self.initializer = tf.contrib.layers.xavier_initializer() 

        self.init_baseline = 0.
        self.is_train = is_train
 
    def predict_rewards_ev(self, i_list):
        frame = i_list
        with tf.variable_scope("ffn_ev"): #, reuse=tf.AUTO_REUSE
            W_0 = tf.get_variable('weights_0', [self.num_neurons, 2*self.num_neurons], initializer=self.initializer)
            w1 =tf.get_variable("w1", [2*self.num_neurons, self.num_neurons], initializer=self.initializer)
            w2 = tf.get_variable("w2", [self.num_neurons, 1], initializer=self.initializer)
            b2 = tf.Variable(self.init_baseline, name="b2")

        h0 = tf.einsum('ijk, kl->ijl', frame, W_0)
        h0 = tf.nn.relu(h0)
        h1 = tf.einsum('ijk, kl->ijl', h0, w1)
        h1 = tf.nn.relu(h1)
        h2 = tf.einsum('ijk, kl->ijl', h1, w2)
        h2 = tf.nn.relu(h2)
        self.predictions_ev = tf.squeeze(h2 + b2)
        return self.predictions_ev

    def predict_rewards_ta(self, i_list_1):
        frame = i_list_1
        with tf.variable_scope("ffn_ta"): #, reuse=tf.AUTO_REUSE
            W_0 = tf.get_variable('weights_0', [self.num_neurons, 2*self.num_neurons], trainable=False, initializer=self.initializer)
            w1 =tf.get_variable("w1", [2*self.num_neurons, self.num_neurons], trainable=False, initializer=self.initializer)
            w2 = tf.get_variable("w2", [self.num_neurons, 1], trainable=False, initializer=self.initializer)
            b2 = tf.Variable(self.init_baseline, trainable=False, name="b1")

        h0 = tf.einsum('ijk, kl->ijl', frame, W_0)
        h0 = tf.nn.relu(h0)
        h1 = tf.einsum('ijk, kl->ijl', h0, w1)
        h1 = tf.nn.relu(h1)
        h2 = tf.einsum('ijk, kl->ijl', h1, w2)
        h2 = tf.nn.relu(h2)
        self.predictions_ta = tf.squeeze(h2 + b2)
        return self.predictions_ta