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
 
# Apply multihead attention to a 3d tensor with shape [batch_size, seq_length, n_hidden].
# Attention size = n_hidden should be a multiple of num_head
# Returns a 3d tensor with shape of [batch_size, seq_length, n_hidden]
 
def multihead_attention(inputs, num_units=None, num_heads=16, dropout_rate=0.1, is_training=True):
 
    with tf.variable_scope("multihead_attention", reuse=None):
        
        # Linear projections
        Q = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        K = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        V = tf.layers.dense(inputs, num_units, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
        
        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
 
        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
        
        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
  
        # Activation
        outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
          
        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
               
        # Weighted sum
        outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        
        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]
              
        # Residual connection
        outputs += inputs # [batch_size, seq_length, n_hidden]
              
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
 
    return outputs
 
 
# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
 
def feedforward(inputs, num_units=[2048, 512], is_training=True):
 
    with tf.variable_scope("ffn", reuse=None):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1, "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1, "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)
        
        # Residual connection
        outputs += inputs
        
        # Normalize
        outputs = tf.layers.batch_normalization(outputs, axis=2, training=is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
    
    return outputs
 
 
class TransformerEncoder(object):
 
    def __init__(self, config, is_train):
        self.batch_size = config.batch_size # batch size
        self.max_length = config.max_length # input sequence length (number of cities)
        self.input_dimension = config.input_dimension # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 token
 
        self.input_embed = config.hidden_dim # dimension of embedding space (actor)
        self.num_heads = config.num_heads
        self.num_stacks = config.num_stacks
 
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer
 
        self.is_training = is_train #not config.inference_mode
 
    def encode(self, inputs):
        with tf.variable_scope("embedding"):
          # Embed input sequence
          W_embed =tf.get_variable("weights",[1,self.input_dimension, self.input_embed], initializer=self.initializer)
          self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
          # Batch Normalization
          self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
        
        with tf.variable_scope("stack"):
          # Blocks
          for i in range(self.num_stacks): # num blocks
              with tf.variable_scope("block_{}".format(i)):
                  # Multihead Attention
                  self.enc = multihead_attention(self.enc, num_units=self.input_embed, num_heads=self.num_heads, dropout_rate=0.0, is_training=self.is_training)
                  
                  # Feed Forward
                  self.enc = feedforward(self.enc, num_units=[4*self.input_embed, self.input_embed], is_training=self.is_training)
 
          # Return the output activations [Batch size, Sequence Length, Num_neurons] as tensors.
          self.encoder_output = self.enc ### NOTE: encoder_output is the ref for attention ###
          return self.encoder_output
 
 
 
 
 
