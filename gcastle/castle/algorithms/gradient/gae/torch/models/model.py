# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


class MLP(nn.Module):
    """
    Feed-forward neural networks----MLP

    """

    def __init__(self, input_dim, layers, units, output_dim,
                 activation=None, device=None) -> None:
        super(MLP, self).__init__()
        # self.desc = desc
        self.input_dim = input_dim
        self.layers = layers
        self.units = units
        self.output_dim = output_dim
        self.activation = activation
        self.device = device

        mlp = []
        for i in range(layers):
            input_size = units
            if i == 0:
                input_size = input_dim
            weight = nn.Linear(in_features=input_size,
                               out_features=self.units,
                               bias=True,
                               device=self.device)
            mlp.append(weight)
            if activation is not None:
                mlp.append(activation)
        out_layer = nn.Linear(in_features=self.units,
                              out_features=self.output_dim,
                              bias=True,
                              device=self.device)
        mlp.append(out_layer)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x) -> torch.Tensor:

        x_ = x.reshape(-1, self.input_dim)
        output = self.mlp(x_)

        return output.reshape(x.shape[0], -1, self.output_dim)


class AutoEncoder(nn.Module):

    def __init__(self, d, input_dim, hidden_layers=3, hidden_dim=16,
                 activation=nn.ReLU(), device=None):
        super(AutoEncoder, self).__init__()
        self.d = d
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.device = device

        self.encoder = MLP(input_dim=self.input_dim,
                           layers=self.hidden_layers,
                           units=self.hidden_dim,
                           output_dim=self.hidden_dim,
                           activation=self.activation,
                           device=self.device)
        self.decoder = MLP(input_dim=self.hidden_dim,
                           layers=self.hidden_layers,
                           units=self.hidden_dim,
                           output_dim=self.input_dim,
                           activation=self.activation,
                           device=self.device)

        w = torch.nn.init.uniform_(torch.empty(self.d, self.d,),
                                   a=-0.1, b=0.1)
        self.w = torch.nn.Parameter(w.to(device=self.device))

    def forward(self, x):

        self.w_adj = self._preprocess_graph(self.w)

        out = self.encoder(x)
        out = torch.einsum('ijk,jl->ilk', out, self.w_adj)
        x_est = self.decoder(out)

        mse_loss = torch.square(torch.norm(x - x_est, p=2))


        return mse_loss, self.w_adj

    def _preprocess_graph(self, w_adj):

        return (1. - torch.eye(w_adj.shape[0], device=self.device)) * w_adj
