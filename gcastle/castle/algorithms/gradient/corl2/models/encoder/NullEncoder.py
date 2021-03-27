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


class Null(object):
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  # input sequence length (number of cities)
        self.input_dimension = config.input_dimension  # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
        self.input_embed = config.hidden_dim  # dimension of embedding space (actor)
        self.initializer = tf.contrib.layers.xavier_initializer()  # variables initializer
        self.is_training = is_train  # not config.inference_mode

    def encode(self, inputs):
        self.encoder_output = inputs  ### NOTE: encoder_output is the ref for attention ###
        return self.encoder_output