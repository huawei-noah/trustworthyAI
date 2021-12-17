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

from ..helpers.utils import generate_upper_triangle_indices


class MaskedQuadraticRegression(nn.Module):

    def __init__(self, mask, n_samples, n_nodes, device=None) -> None:
        super(MaskedQuadraticRegression, self).__init__()
        self.mask = mask
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.device = device

        self._init_weight()

    def forward(self, x, choice) -> torch.Tensor:
        """

        Parameters
        ----------
        x: torch.Tensor
            shape = (n, d - 1)

        Returns
        -------
        output: torch.Tensor
            a vector of shape = (n,)
        """

        output = torch.zeros(self.n_samples, device=self.device)

        # Linear terms
        output += torch.sum(self.weight[choice]['w1'] * x, dim=1)

        # Squared terms
        output += torch.sum(self.weight[choice]['w2'] * torch.square(x), dim=1)

        # Cross terms
        x_ = x.unsqueeze(1)
        y_ = x.unsqueeze(2)
        target_indices = generate_upper_triangle_indices(x.shape[1])
        all_cross_terms = torch.reshape(x_ * y_, (self.n_samples, -1))
        combinations_cross_terms = all_cross_terms[:, target_indices]
        self.w3 = torch.nn.init.uniform_(torch.Tensor(len(target_indices), ),
                                         a=-0.05, b=0.05).requires_grad_(True)
        self.w3 = self.w3.to(device=self.device)
        output += torch.sum(self.w3 * combinations_cross_terms, dim=1)

        # # Bias term
        # b = torch.randn(self.n_samples)
        # output += b
        return output

    def _init_weight(self):

        md = {}
        for i in range(self.mask.shape[0]):
            w = {}
            pns_parents = torch.where(self.mask[:, i] == 1)[0]
            first_input_dim = len([int(j) for j in pns_parents if j != i])
            if first_input_dim == 0:
                continue
            w1 = torch.nn.init.uniform_(torch.Tensor(first_input_dim, ),
                                        a=-0.05, b=0.05)
            self.w1 = torch.nn.Parameter(w1.to(device=self.device))
            w2 = torch.nn.init.uniform_(torch.Tensor(first_input_dim, ),
                                        a=-0.05, b=0.05)
            self.w2 = torch.nn.Parameter(w2.to(device=self.device))
            w['w1'] = self.w1
            w['w2'] = self.w2
            md[str(i)] = w

        self.weight = md
