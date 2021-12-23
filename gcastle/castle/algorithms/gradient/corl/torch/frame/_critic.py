# coding = utf-8
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


class EpisodicCritic(nn.Module):
    """"""

    def __init__(self, input_dim, neurons=(512, 256, 1),
                 activation=nn.ReLU(), device=None) -> None:
        super(EpisodicCritic, self).__init__()
        self.input_dim = input_dim
        self.neurons = neurons
        self.output_dim = neurons[-1]
        self.hidden_units = neurons[:-1]
        self.activation = activation
        self.device = device

        # trainable parameters
        env_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device)
            )
        self.env_w0 = nn.Parameter(env_w0.requires_grad_(True))

        env_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device)
            )
        self.env_w1 = nn.Parameter(env_w1.requires_grad_(True))

        env_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device)
            )
        self.env_w2 = nn.Parameter(env_w2.requires_grad_(True))

        env_b1 = torch.tensor([0.], requires_grad=True, device=self.device)
        self.env_b1 = nn.Parameter(env_b1)

        # Un-trainable parameters
        self.tgt_w0 = nn.init.xavier_uniform_(
            torch.empty(self.input_dim, self.neurons[0], device=self.device)
            )
        self.tgt_w1 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[0], self.neurons[1], device=self.device)
            )
        self.tgt_w2 = nn.init.xavier_uniform_(
            torch.empty(self.neurons[1], self.neurons[-1], device=self.device)
            )
        self.tgt_b1 = torch.tensor([0.], device=self.device)

    def predict_env(self, stats_x) -> None:
        """predict environment reward"""

        stats_x = stats_x.detach()
        h0 = torch.einsum('ijk, kl->ijl', stats_x, self.env_w0)
        h0 = self.activation(h0)
        h1 = torch.einsum('ijk, kl->ijl', h0, self.env_w1)
        h1 = self.activation(h1)
        h2 = torch.einsum('ijk, kl->ijl', h1, self.env_w2)
        h2 = self.activation(h2)

        # [batch_size, seq_length - 1]
        self.prediction_env = (h2 + self.env_b1).squeeze()

    def predict_tgt(self, stats_y) -> None:
        """predict target reward"""

        stats_y = stats_y.detach()
        h0 = torch.einsum('ijk, kl->ijl', stats_y, self.tgt_w0)
        h0 = self.activation(h0)
        h1 = torch.einsum('ijk, kl->ijl', h0, self.tgt_w1)
        h1 = self.activation(h1)
        h2 = torch.einsum('ijk, kl->ijl', h1, self.tgt_w2)
        h2 = self.activation(h2)

        self.prediction_tgt = (h2 + self.tgt_b1).squeeze()

    def soft_replacement(self) -> None:
        # soft_replacement
        self.tgt_w0 = 0.95 * self.tgt_w0 + 0.05 * self.env_w0.detach()
        self.tgt_w1 = 0.95 * self.tgt_w1 + 0.05 * self.env_w1.detach()
        self.tgt_w2 = 0.95 * self.tgt_w2 + 0.05 * self.env_w2.detach()
        self.tgt_b1 = 0.95 * self.tgt_b1 + 0.05 * self.env_b1.detach()


class DenseCritic(nn.Module):
    """Critic network for `dense reward` type

    Only one layer network.
    """

    def __init__(self, input_dim, output_dim, device=None) -> None:
        super(DenseCritic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.h0 = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim,
                      device=device),
            nn.ReLU().to(device=device)
        )
        self.w1 = torch.rand(self.output_dim, 1,
                             device=device).requires_grad_(True)
        self.b1 = torch.tensor([0.], requires_grad=True, device=device)
        self.params = nn.ParameterList([nn.Parameter(self.w1),
                                        nn.Parameter(self.b1)])

    def predict_reward(self, encoder_output) -> torch.Tensor:
        """Predict reward for `dense reward` type"""

        frame = torch.mean(encoder_output, 1).detach()
        h0 = self.h0(frame)
        prediction = torch.matmul(h0, self.w1) + self.b1

        return prediction.squeeze()

