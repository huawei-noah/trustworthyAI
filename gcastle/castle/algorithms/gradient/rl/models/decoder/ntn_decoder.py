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


class NTNDecoder(object):

    def __init__(self, config, is_train):
        self.batch_size = config.batch_size    # batch size
        self.max_length = config.max_length    # input sequence length (number of cities)
        self.input_dimension = config.hidden_dim
        self.input_embed = config.hidden_dim    # dimension of embedding space (actor)
        self.max_length = config.max_length
        self.decoder_hidden_dim = config.decoder_hidden_dim
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.decoder_activation = config.decoder_activation
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
        with tf.variable_scope('ntn'):
            W = tf.get_variable('bilinear_weights', [self.input_embed, self.input_embed, self.decoder_hidden_dim],
                                initializer=self.initializer)
            W_l = tf.get_variable('weights_left', [self.input_embed, self.decoder_hidden_dim],
                                  initializer=self.initializer)
            W_r = tf.get_variable('weights_right', [self.input_embed, self.decoder_hidden_dim],
                                  initializer=self.initializer)
            U = tf.get_variable('U', [self.decoder_hidden_dim], initializer=self.initializer)
            B = tf.get_variable('bias', [self.decoder_hidden_dim], initializer=self.initializer)

        # Compute linear output with shape (batch_size, max_length, max_length, decoder_hidden_dim)
        dot_l = tf.einsum('ijk, kl->ijl', encoder_output, W_l)
        dot_r = tf.einsum('ijk, kl->ijl', encoder_output, W_r)
        tiled_l = tf.tile(tf.expand_dims(dot_l, axis=2), (1, 1, self.max_length, 1))
        tiled_r = tf.tile(tf.expand_dims(dot_r, axis=1), (1, self.max_length, 1, 1))
        linear_sum = tiled_l + tiled_r

        # Compute bilinear product with shape (batch_size, max_length, max_length, decoder_hidden_dim)
        bilinear_product = tf.einsum('ijk, knl, imn->ijml', encoder_output, W, encoder_output)

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = tf.nn.tanh(bilinear_product + linear_sum + B)
        elif self.decoder_activation == 'relu':
            final_sum = tf.nn.relu(bilinear_product + linear_sum + B)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = bilinear_product + linear_sum + B
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        logits = tf.einsum('ijkl, l->ijk', final_sum, U)    # Readability

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
