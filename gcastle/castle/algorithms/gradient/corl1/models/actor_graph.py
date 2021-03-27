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

import logging
import tensorflow as tf
import numpy as np
import copy

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

        self.gamma = 0.98
        # Training config (actor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")  # global step
        self.lr1 = 0.0001
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2")  # global step
        self.lr2 = 0.001

        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension],
                                     name="input_coordinates")
        self.input_true_order_ = tf.placeholder(tf.int32, [self.batch_size, self.max_length],
                                                name="input_order")
        self.reward_ = tf.placeholder(tf.float32, [self.batch_size], name='input_rewards')
        self.reward_list_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length], name='input_reward_list')
        self.target_values_ = tf.placeholder(tf.float32, [self.max_length, self.batch_size], name='target_values_')
        self.i_list_ev = tf.placeholder(tf.float32, [self.max_length - 1, self.batch_size, self.input_dimension],
                                        name='input_critic_eva')
        self.i_list_ta = tf.placeholder(tf.float32, [self.max_length - 1, self.batch_size, self.input_dimension],
                                        name='input_critic_tar')

        self.prev_state_0 = tf.placeholder(tf.float32, [self.max_length*self.batch_size, self.input_dimension],
                                        name='prev_state')
        self.prev_state_1 = tf.placeholder(tf.float32, [self.max_length * self.batch_size, self.input_dimension],
                                           name='prev_state')
        self.prev_input = tf.placeholder(tf.float32, [self.max_length*self.batch_size, self.input_dimension],
                                        name='prev_input')
        self.position = tf.placeholder(tf.float32, [self.max_length*self.batch_size],
                                        name='position')
        self.action_mask_ = tf.placeholder(tf.float32, [self.max_length * self.batch_size, self.max_length],
                                       name='action_mask_')

        self.build_permutation()
        self.build_critic()

        self.build_reward()
        #self.build_reward_func()
        self.build_optim()
        self.merged = tf.summary.merge_all()

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
                self.log_softmax = tf.transpose(tf.reshape(log_softmax, [self.batch_size, self.max_length]),[1,0])
            else:
                raise ('Please choose decoder')

    def build_critic(self):
        with tf.variable_scope("critic"):
            # Critic predicts reward (parametric baseline for REINFORCE)
            self.critic = Critic(self.config, self.is_train)
            self.critic.predict_rewards_ev(self.i_list_ev)
            self.critic.predict_rewards_ta(self.i_list_ta)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/ffn_ev')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic/ffn_ta')

            self.soft_replacement = [tf.assign(t, (1 - 0.05) * t + 0.05 * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def build_reward(self):
        with tf.name_scope('environment'):
            self.reward = self.reward_
            self.reward_list = self.reward_list_

    def build_optim(self):
        with tf.name_scope('reinforce'):
            # Optimizer
            self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1, beta1=0.9, beta2=0.99, epsilon=0.0000001)
            # print('self.reward_list.shape:', self.reward_list.shape)  # (64,12)
            self.advantages = []
            self.reward_baselines = []

            td_target = self.target_values_[:-1]  #(max_length, batch)
            self.advantage = td_target - self.critic.predictions_ev

            # assert advantage.shape == (self.max_length-1,self.batch_size,)
            advantage_st = tf.stop_gradient(self.advantage)
            step_loss = advantage_st * self.log_softmax[:-1]
            # assert step_loss.shape == (self.max_length-1,self.batch_size,)

            # Loss
            self.loss1 = - tf.reduce_mean(step_loss)
            # Minimize step
            gvs = self.opt1.compute_gradients(self.loss1)
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip
            self.train_step1 = self.opt1.apply_gradients(capped_gvs, global_step=self.global_step)

        with tf.name_scope('state_value'):
            # Optimizer
            self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr2, beta1=0.9, beta2=0.99, epsilon=0.0000001)

            td_error = tf.squeeze(tf.reshape(self.advantage, [-1, 1]))

            self.loss2 = tf.reduce_mean(tf.square(td_error))

            # Minimize step
            gvs2 = self.opt2.compute_gradients(self.loss2)
            capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None]  # L2 clip
            self.train_step2 = self.opt2.apply_gradients(capped_gvs2, global_step=self.global_step2)

