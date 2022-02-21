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

    def __init__(self, batch_size, max_length, input_dimension, hidden_dim,
                 init_baseline, device=None):
        super().__init__()

        self.batch_size      = batch_size
        self.max_length      = max_length
        self.input_dimension = input_dimension
        # Network config
        self.input_embed     = hidden_dim
        self.num_neurons     = hidden_dim
        self.device     = device

        # Baseline setup
        self.init_baseline = init_baseline

        layer0 = nn.Linear(in_features=self.input_dimension,
                           out_features=self.num_neurons).to(self.device)
        torch.nn.init.xavier_uniform_(layer0.weight)
        self.h0_layer = nn.Sequential(layer0, nn.ReLU()).to(self.device)

        self.layer1 = nn.Linear(in_features=self.num_neurons,
                                out_features=1).to(self.device)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        self.layer1.bias.data = torch.Tensor([self.init_baseline]).to(self.device)

    def forward(self, encoder_output):
        # [Batch size, Sequence Length, Num_neurons] to [Batch size, Num_neurons]
        frame = torch.mean(encoder_output.detach(), dim=1)
 
        # ffn 1
        h0 = self.h0_layer(frame)
        # ffn 2
        h1 = self.layer1(h0)
        self.predictions = torch.squeeze(h1)
