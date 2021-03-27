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


class Mlp_decoder(object):
    '''
    RNN decoder for pointer network
    '''

    def __init__(self, encoder_output, config, input_true_order_):
        self.batch_size = config.batch_size    # batch size
        self.max_length = config.max_length
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
        self.use_bias = config.use_bias

        self.seq_length = encoder_output.get_shape().as_list()[1]  # sequence length
        self.n_hidden = encoder_output.get_shape().as_list()[2]  # num_neurons
        self.encoder_output = encoder_output  # Tensor [Batch size x time steps x cell.state_size] to attend to
        self.encoder_output_ex = tf.reshape(tf.tile(tf.expand_dims(encoder_output, 1), [1, self.seq_length, 1, 1]),
                                            [-1, self.seq_length, self.n_hidden])
        self.h = tf.transpose(self.encoder_output, [1, 0, 2])

        self.C = config.C  # logit clip
        self.decoder_first_input = tf.reduce_mean(self.encoder_output, 1)
        self.log_softmax = []  # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = []  # store visited cities for reward
        self.mask = 0
        self.mask_scores = []
        # Attending mechanism
        with tf.variable_scope("glimpse") as glimpse:
            self.W_ref_g = tf.get_variable("W_ref_g", [1, self.n_hidden, self.n_hidden], initializer=self.initializer)
            self.W_q_g = tf.get_variable("W_q_g", [self.n_hidden, self.n_hidden], initializer=self.initializer)
            self.v_g = tf.get_variable("v_g", [self.n_hidden], initializer=self.initializer)

        with tf.variable_scope("pointer") as pointer:
            self.W_ref =tf.get_variable("W_ref",[1,self.n_hidden,self.n_hidden],initializer=self.initializer)
            self.W_q =tf.get_variable("W_q",[self.n_hidden,self.n_hidden],initializer=self.initializer)
            self.v =tf.get_variable("v",[self.n_hidden],initializer=self.initializer)

    def mlp(self, inputs):
        num_units = [2048, 512]
        with tf.variable_scope("ffn", reuse=None):
            # # Inner layer
            # params = {"inputs": inputs, "filters": 2* num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
            # outputs = tf.layers.conv1d(**params)
            # # Readout layer
            # params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
            # outputs = tf.layers.conv1d(**params)
            # # Normalize
            # outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, self.n_hidden, activation=None)
        return outputs

    def decode_softmax(self,prev_state_0, prev_state_1, prev_input, position, mask):
        with tf.variable_scope("loop"):
            tf.get_variable_scope().reuse_variables()
            output = self.mlp(prev_input)
            self.mask = mask
            masked_scores = self.attention(self.encoder_output_ex, output)  # [batch_size, time_sequence]

            prob = distr.Categorical(masked_scores)
            log_softmax = prob.log_prob(position)

            return log_softmax


    def attention(self, ref, query):

        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, self.W_ref_g, 1, "VALID",
                                     name="encoded_ref_g")  # [Batch size, seq_length, n_hidden]
        encoded_query_g = tf.expand_dims(tf.matmul(query, self.W_q_g, name="encoded_query_g"),
                                         1)  # [Batch size, 1, n_hidden]
        scores_g = tf.reduce_sum(self.v_g * tf.tanh(encoded_ref_g + encoded_query_g), [-1],
                                 name="scores_g")  # [Batch size, seq_length]

        attention_g = tf.nn.softmax(scores_g - 100000000. * self.mask, name="attention_g")  ###########

        glimpse = tf.multiply(ref, tf.expand_dims(attention_g, 2))  # 64*12*64 64*12*1
        glimpse = tf.reduce_sum(glimpse, 1) + query  ########### Residual connection

        # Pointing mechanism with 1 glimpse
        encoded_ref = tf.nn.conv1d(ref, self.W_ref, 1, "VALID",
                                   name="encoded_ref")  # [Batch size, seq_length, n_hidden]
        encoded_query = tf.expand_dims(tf.matmul(glimpse, self.W_q, name="encoded_query"),
                                       1)  # [Batch size, 1, n_hidden]
        scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1],
                               name="scores")  # [Batch size, seq_length]
        scores = self.C * tf.tanh(scores)  # control entropy

        # Point to cities to visit only (Apply mask)
        masked_scores = scores - 100000000. * self.mask  # [Batch size, seq_length]

        return masked_scores

    # One pass of the decode mechanism
    def decode(self, prev_input, timestep):
        with tf.variable_scope("loop"):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # Run the cell on a combination of the previous input and state
            output = self.mlp(prev_input)
            # Attention mechanism
            masked_scores = self.attention(self.encoder_output, output)  # [batch_size, time_sequence]
            # Multinomial distribution
            prob = distr.Categorical(masked_scores)
            # Sample from distribution
            position = prob.sample()
            position = tf.cast(position, tf.int32)

            self.positions.append(position)
            self.log_softmax.append(prob.log_prob(position))

            # Update mask
            self.mask = self.mask + tf.one_hot(position, self.seq_length)
            self.mask_scores.append(masked_scores)

            # Retrieve decoder's new input
            new_decoder_input = tf.gather(self.h, position)[0]
            return new_decoder_input

    def loop_decode(self):
        i = tf.cast(self.decoder_first_input, tf.float32)
        i_list = []
        for step in range(self.seq_length):
            i_list.append(i)
            i = self.decode(i, step)
        self.i_list = tf.stack(i_list, axis=1)  # [Batch,seq_length,hidden]

        # Stack visited indices
        self.positions = tf.stack(self.positions, axis=1)  # [Batch,seq_length]
        self.mask_scores = tf.stack(self.mask_scores, axis=1)  # [Batch,seq_length]
        self.log_softmax = tf.add_n(self.log_softmax)  # [Batch,seq_length] TODO:[Batch,]
        #self.log_softmax_first_run = tf.add_n(self.log_softmax_first_run)  # TODO:[Batch,]

        # Return stacked lists of visited_indices and log_softmax for backprop
        return self.positions, self.mask_scores, self.i_list, self.i_list, self.i_list

