# coding = utf-8
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
    def __init__(self, input_dim, hidden_dim, output_dim, device=None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        self.w1 = nn.Linear(in_features=self.input_dim,
                            out_features=self.hidden_dim,
                            bias=True,
                            device=self.device)
        self.w2 = nn.Linear(in_features=self.hidden_dim,
                            out_features=self.output_dim,
                            bias=True,
                            device=self.device)

    def forward(self, x):
        out_x = self.w1(x)
        out_x = torch.relu(out_x)
        out_x = self.w2(out_x)

        return out_x

################################
#===========Encoder=============
################################
class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, adj_A, device=None, encoder_type='mlp'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        self.encoder_type = encoder_type
        if self.encoder_type == 'mlp':
            self.mlp = MLP(input_dim, hidden_dim, output_dim, device=device)

        # =====initialize encoder hyper-parameters=====
        self.adj_A = nn.Parameter(adj_A, requires_grad=True)
        self.wa = nn.Parameter(torch.zeros(self.output_dim, device=self.device),
                               requires_grad=True)
        self.z = nn.Parameter(torch.tensor(0.1, device=self.device),
                              requires_grad=True)
        self.z_positive = nn.Parameter(torch.ones_like(self.adj_A, device=self.device),
                                       requires_grad=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3. * self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = torch.eye(adj_A1.shape[0], device=self.device) - adj_A1.T
        if self.encoder_type == 'mlp':
            mlp_out = self.mlp(x.to(self.mlp.device))
            logits = torch.matmul(adj_Aforz, mlp_out + self.wa) - self.wa
        else:
            adj_A_inv = torch.inverse(adj_Aforz)
            meanF = torch.matmul(adj_A_inv, torch.mean(torch.matmul(adj_Aforz, x), 0))
            logits = torch.matmul(adj_Aforz, x - meanF)

        return (logits, adj_A1)


################################
#===========Decoder=============
################################
class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, device=None, decoder_type='mlp'):
        super(Decoder, self).__init__()
        self.decoder_type = decoder_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        if self.decoder_type == 'mlp':
            self.mlp = MLP(input_dim, hidden_dim, output_dim, device=device)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, z, adj_A, wa):

        # adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = torch.inverse(torch.eye(adj_A.shape[0], device=self.device) - adj_A.T)
        mat_z = torch.matmul(adj_A_new1, z + wa) - wa
        if self.decoder_type == 'mlp':
            out = self.mlp(mat_z)
        else:
            out = mat_z
        return out
