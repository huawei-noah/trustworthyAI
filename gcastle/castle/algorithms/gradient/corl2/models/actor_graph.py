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
import numpy as np

from .encoder import TransformerEncoder, MLPEncoder, LSTMEncoder, Null
from .decoder import Pointer_decoder, Mlp_decoder
from .critic import Critic


class Actor(object):

    def __init__(self, config):
        self.config = config
        self.is_train = True
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length
        self.input_dimension = config.input_dimension

        # Reward config
        self.alpha = config.alpha  # moving average update
        self.avg_baseline = tf.Variable(config.init_baseline, trainable=False,
                                        name="moving_avg_baseline")  # moving baseline for Reinforce
        self.gamma = 0.98
        # Training config (actor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")  # global step
        self.lr1 = 0.0001
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2")  # global step
        self.lr2 = 0.001

        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension],
                                     name="input_coordinates")
        self.reward_ = tf.placeholder(tf.float32, [self.batch_size], name='input_rewards')
        self.input_true_order_ = tf.placeholder(tf.int32, [self.batch_size, self.max_length],
                                                name="input_order")
        self.prev_state_0 = tf.placeholder(tf.float32, [self.max_length * self.batch_size, self.input_dimension],
                                           name='prev_state')
        self.prev_state_1 = tf.placeholder(tf.float32, [self.max_length * self.batch_size, self.input_dimension],
                                           name='prev_state')
        self.prev_input = tf.placeholder(tf.float32, [self.max_length * self.batch_size, self.input_dimension],
                                         name='prev_input')
        self.position = tf.placeholder(tf.float32, [self.max_length * self.batch_size],
                                       name='position')
        self.action_mask_ = tf.placeholder(tf.float32, [self.max_length * self.batch_size, self.max_length],
                                           name='action_mask_')

        self.build_permutation()
        self.build_critic()
        self.build_reward()
        self.build_optim()

    def build_permutation(self):
        with tf.variable_scope("encoder"):
            if self.config.encoder_type == 'TransformerEncoder':
                encoder = TransformerEncoder(self.config, self.is_train)
            elif self.config.encoder_type == 'MLPEncoder':
                encoder = MLPEncoder(self.config, self.is_train)
            elif self.config.encoder_type == 'LSTMEncoder':
                encoder = LSTMEncoder(self.config, self.is_train)
            elif self.config.encoder_type == 'Null':
                encoder = Null(self.config, self.is_train)
            else:
                raise NotImplementedError('Current encoder type is not implemented yet!')
            self.encoder_output = encoder.encode(self.input_)

        with tf.variable_scope('decoder'):
            if self.config.decoder_type == 'MLPDecoder':
                self.decoder = Mlp_decoder(self.encoder_output, self.config, self.input_true_order_)
            elif self.config.decoder_type == 'PointerDecoder':
                self.decoder = Pointer_decoder(self.encoder_output, self.config, self.input_true_order_)
            else:
                raise NotImplementedError('Current decoder type is not implemented yet!')

            if self.config.decoder_type == 'PointerDecoder' or 'MLPDecoder':
                self.positions, self.mask_scores, self.s0_list, self.s1_list, self.i_list = self.decoder.loop_decode()
                log_softmax = self.decoder.decode_softmax(self.prev_state_0, self.prev_state_1, self.prev_input, self.position, self.action_mask_)
                self.log_softmax_ = tf.transpose(tf.reshape(log_softmax, [self.batch_size, self.max_length]),[1,0])
                self.log_softmax = tf.reduce_sum(self.log_softmax_, 0)  # TODO:[Batch,]
                assert self.log_softmax.shape == (self.batch_size,)

    def build_critic(self):
        with tf.variable_scope("critic"):
            # Critic predicts reward (parametric baseline for REINFORCE)
            self.critic = Critic(self.config, self.is_train)
            self.critic.predict_rewards(self.encoder_output)

    def build_reward(self):
        with tf.name_scope('environment'):
            self.reward = self.reward_

    def build_optim(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('baseline'):
                reward_mean, reward_var = tf.nn.moments(self.reward, axes=[0])
                self.reward_batch = reward_mean
                self.base_op = tf.assign(self.avg_baseline,
                                         self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean)

            with tf.name_scope('reinforce'):
                # Actor learning rate
                self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1, beta1=0.9, beta2=0.99, epsilon=0.0000001)
                self.reward_baseline = tf.stop_gradient(
                     self.reward - self.avg_baseline - self.critic.predictions)  # [Batch size, 1]

                self.loss1 = - tf.reduce_mean(self.reward_baseline * self.log_softmax, 0)
                # Minimize step
                gvs = self.opt1.compute_gradients(self.loss1)
                capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip
                self.train_step1 = self.opt1.apply_gradients(capped_gvs, global_step=self.global_step)

            with tf.name_scope('state_value'):
                # Optimizer
                self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr2, beta1=0.9, beta2=0.99, epsilon=0.0000001)
                self.loss2 = tf.losses.mean_squared_error(self.reward - self.avg_baseline, self.critic.predictions,
                                                          weights=1.0)
                # Minimize step
                gvs2 = self.opt2.compute_gradients(self.loss2)
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None]  # L2 clip
                self.train_step2 = self.opt2.apply_gradients(capped_gvs2, global_step=self.global_step2)


