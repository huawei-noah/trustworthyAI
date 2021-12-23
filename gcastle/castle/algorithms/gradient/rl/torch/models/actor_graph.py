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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .encoder import TransformerEncoder, GATEncoder
from .decoder import TransformerDecoder, SingleLayerDecoder, BilinearDecoder, NTNDecoder
from .critic import Critic


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
        self.avg_baseline = torch.Tensor([config.init_baseline])  # moving baseline for Reinforce
        self.alpha = config.alpha  # moving average update

        # Training config (actor)
        self.global_step = torch.Tensor([0])  # global step
        self.lr1_start = config.lr1_start  # initial learning rate
        self.lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr1_decay_step = config.lr1_decay_step  # learning rate decay step

        # Training config (critic)
        self.global_step2 = torch.Tensor([0])  # global step
        self.lr2_start = config.lr1_start  # initial learning rate
        self.lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
        self.lr2_decay_step = config.lr1_decay_step  # learning rate decay step

        # encoder
        if self.config.encoder_type == 'TransformerEncoder':
            self.encoder = TransformerEncoder(self.config, self.is_train)
        elif self.config.encoder_type == 'GATEncoder':
            self.encoder = GATEncoder(self.config, self.is_train)
        else:
            raise NotImplementedError('Current encoder type is not implemented yet!')

        # decoder
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

        # critic
        self.critic = Critic(self.config, self.is_train)
        
        # Optimizer
        self.opt1 = torch.optim.Adam([
                        {'params': self.encoder.parameters()},
                        {'params': self.decoder.parameters()},
                        {'params': self.critic.parameters()}
                    ], lr=self.lr1_start, betas=(0.9, 0.99), eps=0.0000001)

        self.lr1_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.opt1, gamma=pow(self.lr1_decay_rate, 1/self.lr1_decay_step))
        
        self.criterion = nn.MSELoss()

    def build_permutation(self, inputs):

        # Tensor block holding the input sequences [Batch Size, Sequence Length, Features]
        self.input_ = inputs

        # encoder
        self.encoder_output = self.encoder(self.input_)

        # decoder
        self.samples, self.scores, self.entropy = self.decoder(self.encoder_output)

        # self.samples is seq_lenthg * batch size * seq_length
        # cal cross entropy loss * reward
        graphs_gen = torch.stack(self.samples).permute([1,0,2])
        # graphs_gen.requires_grad = True
        self.graphs_ = graphs_gen

        self.graph_batch = torch.mean(graphs_gen, axis=0)
        logits_for_rewards = torch.stack(self.scores)
        # logits_for_rewards.requires_grad = True
        entropy_for_rewards = torch.stack(self.entropy)
        # entropy_for_rewards.requires_grad = True
        entropy_for_rewards = entropy_for_rewards.permute([1, 0, 2])
        logits_for_rewards = logits_for_rewards.permute([1, 0, 2])
        self.test_scores = torch.sigmoid(logits_for_rewards)[:2]
        log_probss = F.binary_cross_entropy_with_logits(input=logits_for_rewards, 
                                                        target=self.graphs_, 
                                                        reduction='none')

        self.log_softmax = torch.mean(log_probss, axis=[1, 2])
        self.entropy_regularization = torch.mean(entropy_for_rewards, axis=[1,2])

        self.build_critic()

    def build_critic(self):
        # Critic predicts reward (parametric baseline for REINFORCE)
        self.critic = Critic(self.config, self.is_train)
        self.critic(self.encoder_output)

    def build_reward(self, reward_):

        self.reward = reward_

        self.build_optim()

    def build_optim(self):
        # Update moving_mean and moving_variance for batch normalization layers
        # Update baseline
        reward_mean, reward_var = torch.mean(self.reward), torch.std(self.reward)
        self.reward_batch = reward_mean
        self.avg_baseline = self.alpha * self.avg_baseline + (1.0 - self.alpha) * reward_mean
        if self.config.device_type == 'gpu':
            self.avg_baseline = self.avg_baseline.cuda(self.config.device_ids)

        # Discounted reward
        self.reward_baseline = (self.reward - self.avg_baseline - self.critic.predictions).detach()  # [Batch size, 1]

        # Loss
        self.loss1 = (torch.mean(self.reward_baseline * self.log_softmax, 0) 
                      - 1*self.lr1_scheduler.get_last_lr()[0] * torch.mean(self.entropy_regularization, 0))
        self.loss2 = self.criterion(self.reward - self.avg_baseline, self.critic.predictions)

        # Minimize step
        self.opt1.zero_grad()
        self.loss1.backward()
        self.loss2.backward()
        
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.critic.parameters()), max_norm=1., norm_type=2)

        self.opt1.step()
        self.lr1_scheduler.step()
