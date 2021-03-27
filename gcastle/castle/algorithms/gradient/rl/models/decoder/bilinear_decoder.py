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
from tensorflow.contrib import distributions as distr


class BilinearDecoder(object):

    def __init__(self, config, is_train):
        self.batch_size = config.batch_size    # batch size
        self.max_length = config.max_length    # input sequence length (number of cities)
        self.input_dimension = config.hidden_dim
        self.input_embed = config.hidden_dim    # dimension of embedding space (actor)
        self.max_length = config.max_length
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.use_bias = config.use_bias
        self.bias_initial_value = config.bias_initial_value
        self.use_bias_constant = config.use_bias_constant
        self.is_training = is_train

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

    def decode(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        with tf.variable_scope('bilinear'):
            W = tf.get_variable('bilinear_weights', [self.input_embed, self.input_embed], initializer=self.initializer)

        logits = tf.einsum('ijk, kn, imn->ijm', encoder_output, W, encoder_output)    # Readability

        if self.bias_initial_value is None:    # Randomly initialize the learnable bias
            self.logit_bias = tf.get_variable('logit_bias', [1])
        elif self.use_bias_constant:    # Constant bias
            self.logit_bias =  tf.constant([self.bias_initial_value], tf.float32, name='logit_bias')
        else:    # Learnable bias with initial value
            self.logit_bias =  tf.Variable([self.bias_initial_value], tf.float32, name='logit_bias')

        if self.use_bias:    # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        for i in range(self.max_length):
            position = tf.ones([encoder_output.shape[0]]) * i
            position = tf.cast(position, tf.int32)

            # Update mask
            self.mask = tf.one_hot(position, self.max_length)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(masked_score)    # probs input probability, logit input log_probability

            sampled_arr = prob.sample()    # Batch_size, seqlenght for just one node

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy
