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

import torch
import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, config, is_train):

        super().__init__()

        self.config = config

        self.device = torch.device("cuda" if config.device_type=='gpu' else "cpu")

        # Data config
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.input_dimension = config.input_dimension

        # Network config
        self.input_embed = config.hidden_dim
        self.num_neurons = config.hidden_dim

        # Baseline setup
        self.init_baseline = 0.

        if self.config.device_type == 'gpu':
            layer0 = nn.Linear(in_features=self.input_dimension, out_features=self.num_neurons).cuda(self.config.device_ids)
            torch.nn.init.xavier_uniform(layer0.weight)
            self.h0_layer = nn.Sequential(layer0, nn.ReLU()).cuda(self.config.device_ids)

            self.layer1 = nn.Linear(in_features=self.num_neurons, out_features=1).cuda(self.config.device_ids)
            torch.nn.init.xavier_uniform(self.layer1.weight)
            self.layer1.bias.data = torch.Tensor([self.init_baseline]).cuda(self.config.device_ids)
        else:
            layer0 = nn.Linear(in_features=self.input_dimension, out_features=self.num_neurons)
            torch.nn.init.xavier_uniform(layer0.weight)
            self.h0_layer = nn.Sequential(layer0, nn.ReLU())

            self.layer1 = nn.Linear(in_features=self.num_neurons, out_features=1)
            torch.nn.init.xavier_uniform(self.layer1.weight)
            self.layer1.bias.data = torch.Tensor([self.init_baseline])

    def forward(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = torch.mean(encoder_output.detach(), dim=1)
 
        # ffn 1
        h0 = self.h0_layer(frame)
        # ffn 2
        h1 = self.layer1(h0)
        self.predictions = torch.squeeze(h1)
