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

'''
Adapted from kyubyong park, June 2017.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell, DropoutWrapper
import numpy as np


class LSTMEncoder(object):
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of cities)
        self.input_dimension = config.input_dimension  # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token

        self.input_embed = config.hidden_dim  # dimension of embedding space (actor)

        self.initializer = tf.contrib.layers.xavier_initializer()  # variables initializer

        self.is_training = is_train  # not config.inference_mode


    def encode(self, inputs):
        # Tensor blocks holding the input sequences [Batch Size, Sequence Length, Features]
        # self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension], name="input_raw")

        with tf.variable_scope("embedding"):
            # Embed input sequence
            W_embed = tf.get_variable("weights", [1, self.input_dimension, self.input_embed],
                                      initializer=self.initializer)  # +2 for TW feat. here too
            embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
            # Batch Normalization
            embedded_input = tf.layers.batch_normalization(embedded_input, axis=2, training=self.is_training,
                                                           name='layer_norm',
                                                           reuse=None)

        with tf.variable_scope("dynamic_rnn"):
            # Encode input sequence
            cell1 = LSTMCell(self.input_embed,
                             initializer=self.initializer)  # BNLSTMCell(self.num_neurons, self.training) or cell1 = DropoutWrapper(cell1, output_keep_prob=0.9)
            # Return the output activations [Batch size, Sequence Length, Num_neurons] and last hidden state as tensors.
            self.encoder_output, encoder_state = tf.nn.dynamic_rnn(cell1, embedded_input, dtype=tf.float32)

            return self.encoder_output

