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
import torch.nn.functional as F

from ._base_network import BaseEncoder


class LSTMEncoder(BaseEncoder):
    """
    Parameters
    ----------
    input_dim: int
        Number of features of input.
    embed_dim: int
        Number of features of hidden layer.
    """

    def __init__(self, input_dim, embed_dim, device=None) -> None:
        super(LSTMEncoder, self).__init__(input_dim=input_dim,
                                          embed_dim=embed_dim,
                                          device=device)
        self.input_dim = input_dim
        self.hidden_dim = embed_dim
        self.device = device
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            bias=True,
                            batch_first=True
                            ).to(device=device)

    def forward(self, x) -> torch.Tensor:
        """

        Parameters
        ----------
        x:
            [Batch Size, Sequence Length, Features]
        """

        x = x.permute(0, 2, 1)
        output = self.embedding(x).permute(0, 2, 1)
        output, (_, _) = self.lstm(output)

        return output


class MLPEncoder(BaseEncoder):
    """
    Feed-forward neural networks----MLP

    """

    def __init__(self, input_dim, embed_dim, hidden_dim,
                 device=None) -> None:
        super(MLPEncoder, self).__init__(input_dim=input_dim,
                                         embed_dim=embed_dim,
                                         hidden_dim=hidden_dim,
                                         device=device)
        self.input_dim = input_dim
        self.embed_dim = embed_dim  # also is output_dim
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, x) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        output = self.embedding(x)
        output = self.feedforward_conv1d(output)
        output = self.bn(output).permute(0, 2, 1)

        return output


class TransformerEncoder(BaseEncoder):
    """Transformer Encoder"""

    def __init__(self, input_dim, embed_dim, hidden_dim,
                 heads=8, blocks=3, device=None) -> None:
        super(TransformerEncoder, self).__init__(input_dim=input_dim,
                                                 embed_dim=embed_dim,
                                                 hidden_dim=hidden_dim,
                                                 device=device)
        self.input_dim = input_dim
        self.heads = heads
        self.embed_dim = embed_dim  # also is output_dim
        self.hidden_dim = hidden_dim
        self.blocks = blocks
        self.device = device
        self.attention = MultiHeadAttention(input_dim=embed_dim,
                                            output_dim=embed_dim,
                                            heads=heads,
                                            dropout_rate=0.0,
                                            device=device)

    def forward(self, x) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        output = self.embedding(x).permute(0, 2, 1)
        for i in range(self.blocks):
            enc = self.attention(output)
            enc = enc.permute(0, 2, 1)
            output = self.feedforward_conv1d(enc)
            output += enc  # Residual connection
            output = self.bn(output).permute(0, 2, 1)

        return output


class MultiHeadAttention(nn.Module):
    """
    Multi head attention mechanism

    Parameters
    ----------
    input_dim: int
        input dimension
    output_dim: int
        output dimension
    heads: int
        head numbers of multi head attention mechanism
    dropout_rate: float, int
        If not 0, append `Dropout` layer on the outputs of each LSTM layer
        except the last layer. Default 0. The range of dropout is (0.0, 1.0).

    """

    def __init__(self, input_dim, output_dim, heads=8, dropout_rate=0.1,
                 device=None) -> None:
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.dropout_rate = dropout_rate
        self.device = device

        self.w_q = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w_k = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w_v = nn.Sequential(
            nn.Linear(in_features=input_dim,
                      out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.bn = nn.BatchNorm1d(num_features=output_dim, device=device)

    def forward(self, x) -> torch.Tensor:
        Q = self.w_q(x)  # [batch_size, seq_length, n_hidden]
        K = self.w_k(x)
        V = self.w_v(x)

        # Split and concat
        Q_ = torch.cat(torch.split(Q,
                                   split_size_or_sections=Q.shape[2]//self.heads,
                                   dim=2),
                       dim=0)
        K_ = torch.cat(torch.split(K,
                                   split_size_or_sections=K.shape[2]//self.heads,
                                   dim=2),
                       dim=0)
        V_ = torch.cat(torch.split(V,
                                   split_size_or_sections=V.shape[2]//self.heads,
                                   dim=2),
                       dim=0)
        # Multiplication # [num_heads*batch_size, seq_length, seq_length]
        output = torch.matmul(Q_, K_.permute(0, 2, 1))

        # Scale
        output = output / (K_.shape[-1] ** 0.5)

        # Activation  # [num_heads*batch_size, seq_length, seq_length]
        output = F.softmax(output, dim=1)

        # Dropouts
        output = F.dropout(output, p=self.dropout_rate)

        # Weighted sum # [num_heads*batch_size, seq_length, n_hidden/num_heads]
        output = torch.matmul(output, V_)

        # Restore shape
        output = torch.cat(torch.split(output,
                                       split_size_or_sections=output.shape[0]//self.heads,
                                       dim=0),
                           dim=2)  # [batch_size, seq_length, n_hidden]
        # Residual connection
        output += x  # [batch_size, seq_length, n_hidden]
        output = self.bn(output.permute(0, 2, 1)).permute(0, 2, 1)

        return output
