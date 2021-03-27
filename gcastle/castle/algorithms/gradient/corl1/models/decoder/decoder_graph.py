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
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib import distributions as distr
import numpy as np

# RNN decoder for pointer network
class Pointer_decoder(object):

    def __init__(self, encoder_output, config, input_true_order_):

        batch_size = encoder_output.get_shape().as_list()[0] # batch size
        self.seq_length = encoder_output.get_shape().as_list()[1] # sequence length
        n_hidden = encoder_output.get_shape().as_list()[2] # num_neurons

        self.encoder_output = encoder_output  # Tensor [Batch size x time steps x cell.state_size] to attend to
        self.encoder_output_ex = tf.reshape(tf.tile(tf.expand_dims(encoder_output, 1),[1,self.seq_length, 1,1]),
                                            [-1, self.seq_length, n_hidden])
        self.h = tf.transpose(self.encoder_output, [1, 0, 2])  # [time steps x Batch size x cell.state_size]

        self.C = config.C # logit clip

        self.input_true_order = input_true_order_

        # Variables initializer
        initializer = tf.contrib.layers.xavier_initializer()

        # Decoder LSTM cell        
        self.cell = LSTMCell(n_hidden, initializer=initializer)

        # Decoder initial state (tuple) is trainable
        self.decoder_first_input = tf.reduce_mean(self.encoder_output, 1)

        first_state = tf.get_variable("GO_state1",[1,n_hidden], initializer=initializer)
        self.decoder_initial_state = tf.tile(first_state, [batch_size,1]), tf.reduce_mean(self.encoder_output,1)

        # Attending mechanism
        with tf.variable_scope("glimpse") as glimpse:
            self.W_ref_g =tf.get_variable("W_ref_g",[1,n_hidden,n_hidden],initializer=initializer)
            self.W_q_g =tf.get_variable("W_q_g",[n_hidden,n_hidden],initializer=initializer)
            self.v_g =tf.get_variable("v_g",[n_hidden],initializer=initializer)

        # Pointing mechanism
        with tf.variable_scope("pointer") as pointer:
            self.W_ref =tf.get_variable("W_ref",[1,n_hidden,n_hidden],initializer=initializer)
            self.W_q =tf.get_variable("W_q",[n_hidden,n_hidden],initializer=initializer)
            self.v =tf.get_variable("v",[n_hidden],initializer=initializer)

        self.log_softmax = [] # store log(p_theta(pi(t)|pi(<t),s)) for backprop
        self.positions = [] # store visited cities for reward

        self.mask = 0
        self.mask_scores = []

    # From a query (decoder output) [Batch size, n_hidden] and a set of reference (encoder_output) [Batch size, seq_length, n_hidden]
    # predict a distribution over next decoder input
    def attention(self,ref,query):
        # Attending mechanism
        encoded_ref_g = tf.nn.conv1d(ref, self.W_ref_g, 1, "VALID", name="encoded_ref_g") # [Batch size, seq_length, n_hidden]
        encoded_query_g = tf.expand_dims(tf.matmul(query, self.W_q_g, name="encoded_query_g"), 1) # [Batch size, 1, n_hidden]
        scores_g = tf.reduce_sum(self.v_g * tf.tanh(encoded_ref_g + encoded_query_g), [-1], name="scores_g") # [Batch size, seq_length]

        attention_g = tf.nn.softmax(scores_g - 100000000.*self.mask,name="attention_g")  ###########

        # 1 glimpse = Linear combination of reference vectors (defines new query vector)
        glimpse = tf.multiply(ref, tf.expand_dims(attention_g,2)) #64*12*64 64*12*1
        glimpse = tf.reduce_sum(glimpse,1)+query  ########### Residual connection

        # Pointing mechanism with 1 glimpse
        encoded_ref = tf.nn.conv1d(ref, self.W_ref, 1, "VALID", name="encoded_ref") # [Batch size, seq_length, n_hidden]
        encoded_query = tf.expand_dims(tf.matmul(glimpse, self.W_q, name="encoded_query"), 1) # [Batch size, 1, n_hidden]
        scores = tf.reduce_sum(self.v * tf.tanh(encoded_ref + encoded_query), [-1], name="scores") # [Batch size, seq_length]
        masked_scores = scores - 100000000.*self.mask # [Batch size, seq_length]

        return masked_scores

    def decode_softmax(self,prev_state_0, prev_state_1, prev_input, position, mask):
        with tf.variable_scope("loop"):
            tf.get_variable_scope().reuse_variables()

            s = prev_state_0, prev_state_1
            output, state = self.cell(prev_input, s)
            self.mask = mask
            masked_scores = self.attention(self.encoder_output_ex, output)  # [batch_size, time_sequence]

            prob = distr.Categorical(masked_scores)
            log_softmax = prob.log_prob(position)

            return log_softmax

    def decode(self,prev_state,prev_input,timestep):
        with tf.variable_scope("loop"):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()

            # Run the cell on a combination of the previous input and state
            output, state = self.cell(prev_input,prev_state)
            masked_scores = self.attention(self.encoder_output, output)  # [batch_size, time_sequence]
            self.mask_scores.append(masked_scores)

            # Multinomial distribution
            prob = distr.Categorical(masked_scores)
            # Sample from distribution
            position = prob.sample()
            position = tf.cast(position, tf.int32)
            self.positions.append(position)

            # Store log_prob for backprop
            self.log_softmax.append(prob.log_prob(position))

            self.mask = self.mask + tf.one_hot(position, self.seq_length)

            # Retrieve decoder's new input
            new_decoder_input = tf.gather(self.h, position)[0]

            return state, new_decoder_input

    def loop_decode(self):
        # decoder_initial_state: Tuple Tensor (c,h) of size [batch_size x cell.state_size]
        # decoder_first_input: Tensor [batch_size x cell.state_size]

        # Loop the decoding process and collect results
        s, i = self.decoder_initial_state, tf.cast(self.decoder_first_input, tf.float32)
        s0_list = []
        s1_list = []
        i_list = []
        for step in range(self.seq_length):
            s0_list.append(s[0])
            s1_list.append(s[1])
            i_list.append(i)
            s, i = self.decode(s, i, step)

        self.s0_list = tf.stack(s0_list, axis=1)  # [Batch,seq_length,hidden]
        self.s1_list = tf.stack(s1_list, axis=1)  # [Batch,seq_length,hidden]
        self.i_list = tf.stack(i_list, axis=1)  # [Batch,seq_length,hidden]

        # Stack visited indices
        self.positions = tf.stack(self.positions, axis=1)  # [Batch,seq_length]
        self.mask_scores = tf.stack(self.mask_scores, axis=1)  # [Batch,seq_length,seq_length]
        self.log_softmax = tf.stack(self.log_softmax, axis=0)
        #self.log_softmax_first_run = tf.add_n(self.log_softmax_first_run)  # TODO:[Batch,]
        self.mask = 0
        return self.positions, self.mask_scores, self.s0_list, self.s1_list, self.i_list
