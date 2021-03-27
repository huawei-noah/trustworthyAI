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

from .encoder import TransformerEncoder, GATEncoder
from .decoder import TransformerDecoder, SingleLayerDecoder, BilinearDecoder, NTNDecoder
from .critic import Critic


# Tensor summaries for TensorBoard visualization
def variable_summaries(name, var, with_max_min=False):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        if with_max_min == True:
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))


class Actor(object):
    _logger = logging.getLogger(__name__)

    def __init__(self, config):
        self.config = config
        self.is_train = True
        # Data config
        self.batch_size = config.batch_size  # batch size
        self.max_length = config.max_length  
        self.input_dimension = config.input_dimension  

        # Reward config
        self.avg_baseline = tf.Variable(config.init_baseline, trainable=False,
                                        name="moving_avg_baseline")  # moving baseline for Reinforce
        self.alpha = config.alpha  # moving average update

        # Training config (actor)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")  # global step
        self.lr1_start = config.lr1_start  # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step

        # Training config (critic)
        self.global_step2 = tf.Variable(0, trainable=False, name="global_step2")  # global step
        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.input_dimension],
                                     name="input_coordinates")
        self.reward_ = tf.placeholder(tf.float32, [self.batch_size], name='input_rewards')
        self.graphs_ = tf.placeholder(tf.float32, [self.batch_size, self.max_length, self.max_length], name='input_graphs')

        self.build_permutation()
        self.build_critic()
        self.build_reward()
        self.build_optim()
        self.merged = tf.summary.merge_all()

    def build_permutation(self):
        with tf.variable_scope("encoder"):
            if self.config.encoder_type == 'TransformerEncoder':
                encoder = TransformerEncoder(self.config, self.is_train)
            elif self.config.encoder_type == 'GATEncoder':
                encoder = GATEncoder(self.config, self.is_train)
            else:
                raise NotImplementedError('Current encoder type is not implemented yet!')
            self.encoder_output = encoder.encode(self.input_)

        with tf.variable_scope('decoder'):
            if self.config.decoder_type == 'SingleLayerDecoder':
                self.decoder = SingleLayerDecoder(self.config, self.is_train)
            elif self.config.decoder_type == 'TransformerDecoder':
                self.decoder = TransformerDecoder(self.config, self.is_train)
            elif self.config.decoder_type == 'BilinearDecoder':
                self.decoder = BilinearDecoder(self.config, self.is_train)
            elif self.config.decoder_type == 'NTNDecoder':
                self.decoder = NTNDecoder(self.config, self.is_train)
            else:
                raise NotImplementedError('Current decoder type is not implemented yet!')

            self.samples, self.scores, self.entropy = self.decoder.decode(self.encoder_output)

            # self.samples is seq_lenthg * batch size * seq_length
            # cal cross entropy loss * reward
            graphs_gen = tf.transpose(tf.stack(self.samples), [1,0,2])

            self.graphs = graphs_gen
            self.graph_batch = tf.reduce_mean(graphs_gen, axis=0)
            logits_for_rewards = tf.stack(self.scores)
            entropy_for_rewards = tf.stack(self.entropy)
            entropy_for_rewards = tf.transpose(entropy_for_rewards, [1, 0, 2])
            logits_for_rewards = tf.transpose(logits_for_rewards, [1, 0, 2])
            self.test_scores = tf.sigmoid(logits_for_rewards)[:2]
            log_probss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.graphs_, logits=logits_for_rewards)
            self.log_softmax = tf.reduce_mean(log_probss, axis=[1, 2])
            self.entropy_regularization = tf.reduce_mean(entropy_for_rewards, axis=[1,2])

            variable_summaries('log_softmax', self.log_softmax, with_max_min=True)

    def build_critic(self):
        with tf.variable_scope("critic"):
            # Critic predicts reward (parametric baseline for REINFORCE)
            self.critic = Critic(self.config, self.is_train)
            self.critic.predict_rewards(self.encoder_output)

            variable_summaries('predictions', self.critic.predictions, with_max_min=True)

    def build_reward(self):
        with tf.name_scope('environment'):
            self.reward = self.reward_
            variable_summaries('reward', self.reward, with_max_min=True)


    def build_optim(self):
        # Update moving_mean and moving_variance for batch normalization layers
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope('baseline'):
                # Update baseline
                reward_mean, reward_var = tf.nn.moments(self.reward, axes=[0])
                self.reward_batch = reward_mean
                self.base_op = tf.assign(self.avg_baseline,
                                         self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean)
                tf.summary.scalar('average baseline', self.avg_baseline)

            with tf.name_scope('reinforce'):
                # Actor learning rate
                self.lr1 = tf.train.exponential_decay(self.lr1_start, self.global_step, self.lr1_decay_step,
                                                      self.lr1_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt1 = tf.train.AdamOptimizer(learning_rate=self.lr1, beta1=0.9, beta2=0.99, epsilon=0.0000001)
                # Discounted reward
                self.reward_baseline = tf.stop_gradient(
                    self.reward - self.avg_baseline - self.critic.predictions)  # [Batch size, 1]
                variable_summaries('reward_baseline', self.reward_baseline, with_max_min=True)
                # Loss
                self.loss1 = tf.reduce_mean(self.reward_baseline * self.log_softmax, 0) -  1* self.lr1 * tf.reduce_mean(self.entropy_regularization, 0)
                tf.summary.scalar('loss1', self.loss1)
                # Minimize step
                gvs = self.opt1.compute_gradients(self.loss1)
                capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs if grad is not None]  # L2 clip
                self.train_step1 = self.opt1.apply_gradients(capped_gvs, global_step=self.global_step)

            with tf.name_scope('state_value'):
                # Critic learning rate
                self.lr2 = tf.train.exponential_decay(self.lr2_start, self.global_step2, self.lr2_decay_step,
                                                      self.lr2_decay_rate, staircase=False, name="learning_rate1")
                # Optimizer
                self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr2, beta1=0.9, beta2=0.99, epsilon=0.0000001)
                # Loss
                weights_ = 1.0  # weights_ = tf.exp(self.log_softmax-tf.reduce_max(self.log_softmax)) # probs / max_prob
                self.loss2 = tf.losses.mean_squared_error(self.reward - self.avg_baseline, self.critic.predictions,
                                                          weights=weights_)
                tf.summary.scalar('loss2', self.loss2)
                # Minimize step
                gvs2 = self.opt2.compute_gradients(self.loss2)
                capped_gvs2 = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs2 if grad is not None]  # L2 clip
                self.train_step2 = self.opt1.apply_gradients(capped_gvs2, global_step=self.global_step2)
