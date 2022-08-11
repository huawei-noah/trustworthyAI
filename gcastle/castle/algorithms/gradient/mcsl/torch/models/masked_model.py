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

from ..helpers.utils import gumbel_sigmoid
from .masked_nn import MaskedNN
from .masked_quadratic_regression import MaskedQuadraticRegression


class MaskedModel(torch.nn.Module):

    def __init__(self, model_type, n_samples, n_nodes, pns_mask, num_hidden_layers,
                 hidden_dim, l1_graph_penalty, seed, device) -> None:
        super(MaskedModel, self).__init__()
        self.model_type = model_type
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.pns_mask = pns_mask
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.l1_graph_penalty = l1_graph_penalty
        self.seed = seed
        self.device = device

        if self.model_type == 'nn':
            self.masked_model = MaskedNN(mask=self.pns_mask,
                                         num_hidden_layers=self.num_hidden_layers,
                                         hidden_dim=self.hidden_dim,
                                         device=self.device
                                         )
        elif self.model_type == 'qr':   # quadratic regression
            self.masked_model = MaskedQuadraticRegression(
                mask=self.pns_mask,
                n_samples=self.n_samples,
                n_nodes=self.n_nodes,
                device=self.device
            )
        else:
            raise TypeError(f"The argument `model_type` must be one of"
                            f"['nn', 'qr'], but got {self.model_type}.")
        torch.manual_seed(self.seed)
        w = torch.nn.init.uniform_(torch.Tensor(self.n_nodes, self.n_nodes),
                                   a=-1e-10, b=1e-10)
        self.w = torch.nn.Parameter(w.to(device=self.device))

    def forward(self, x, rho, alpha, temperature) -> tuple:

        w_prime = self._preprocess_graph(self.w, tau=temperature,
                                         seed=self.seed)
        w_prime = self.pns_mask * w_prime
        mse_loss = self._get_mse_loss(x, w_prime)
        h = (torch.trace(torch.matrix_exp(w_prime * w_prime)) - self.n_nodes)
        loss = (0.5 / self.n_samples * mse_loss
                + self.l1_graph_penalty * torch.linalg.norm(w_prime, ord=1)
                + alpha * h
                + 0.5 * rho * h * h)

        return loss, h, self.w

    def _preprocess_graph(self, w, tau, seed=0) -> torch.Tensor:

        w_prob = gumbel_sigmoid(w, temperature=tau, seed=seed,
                                device=self.device)
        w_prob = (1. - torch.eye(w.shape[0], device=self.device)) * w_prob

        return w_prob

    def _get_mse_loss(self, x, w_prime):
        """
        Different model for different nodes to use masked features to predict
        value for each node.
        """

        mse_loss = 0
        for i in range(self.n_nodes):
            # Get possible PNS parents and also remove diagonal element
            pns_parents = torch.where(self.pns_mask[:, i] == 1)[0]
            possible_parents = [int(j) for j in pns_parents if j != i]
            if len(possible_parents) == 0:    # Root node, don't have to build NN in this case
                continue
            curr_x = x[:, possible_parents]    # Features for current node
            curr_y = x[:, i]   # Label for current node
            curr_w = w_prime[possible_parents, i]   # Mask for current node
            curr_masked_x = curr_x * curr_w   # Broadcasting
            curr_y_pred = self.masked_model(curr_masked_x, choice=str(i))    # Use masked features to predict value of current node

            mse_loss = mse_loss + torch.sum(torch.square(curr_y_pred.squeeze() - curr_y))

        return mse_loss
