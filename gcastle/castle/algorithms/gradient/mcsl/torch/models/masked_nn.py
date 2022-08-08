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


class MaskedNN(nn.Module):

    def __init__(self, mask, num_hidden_layers, hidden_dim,
                 device=None) -> None:
        super(MaskedNN, self).__init__()
        self.mask = mask    # use mask to determine input dimension
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.device = device

        self._init_nn()

    def forward(self, x, choice) -> torch.Tensor:
        """

        Parameters
        ----------
        x: torch.Tensor
            possible parents
        choice: str of int
            current sub-note y

        Returns
        -------
        output: torch.Tensor
            shape = (n,)
        """

        output = self.nets[choice](x)

        return output

    def _init_nn(self):
        """ Initialize net for each node"""

        md = {}
        for i in range(self.mask.shape[0]):
            pns_parents = torch.where(self.mask[:, i] == 1)[0]
            first_input_dim = len([int(j) for j in pns_parents if j != i])
            if first_input_dim == 0:    # Root node, don't have to build NN in this case
                continue
            reg_nn = []
            for j in range(self.num_hidden_layers):
                input_dim = self.hidden_dim
                if j == 0:
                    input_dim = first_input_dim
                func = nn.Sequential(
                    nn.Linear(in_features=input_dim,
                          out_features=self.hidden_dim).to(device=self.device),
                    nn.LeakyReLU(negative_slope=0.05).to(device=self.device)
                )
                reg_nn.append(func)
            output_layer = nn.Linear(in_features=self.hidden_dim,
                                     out_features=1).to(device=self.device)
            reg_nn.append(output_layer)
            reg_nn = nn.Sequential(*reg_nn)

            md[str(i)] = reg_nn
        self.nets = nn.ModuleDict(md)
