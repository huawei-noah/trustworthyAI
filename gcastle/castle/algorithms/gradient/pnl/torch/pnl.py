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

import os
import itertools
import logging
import torch
import torch.nn as nn
import numpy as np

from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import hsic_test

from .utils import batch_loader, compute_jacobian, compute_entropy


class MLP(nn.Module):
    """
    Multi-layer perceptron

    """

    def __init__(self, input_dim, hidden_layers, hidden_units, output_dim,
                 bias=True, activation=None, device=None) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.bias = bias
        self.activation = activation
        self.device = device

        mlp = []
        for i in range(self.hidden_layers):
            input_size = self.hidden_units
            if i == 0:
                input_size = self.input_dim
            weight = nn.Linear(in_features=input_size,
                               out_features=self.hidden_units,
                               bias=self.bias,
                               device=self.device)
            mlp.append(weight)
            if self.activation is not None:
                mlp.append(self.activation)
        out_layer = nn.Linear(in_features=self.hidden_units,
                              out_features=self.output_dim,
                              bias=self.bias,
                              device=self.device)
        mlp.append(out_layer)

        self.mlp = nn.Sequential(*mlp)

    def forward(self, x) -> torch.Tensor:

        out = self.mlp(x)

        return out


class PNL(BaseLearner):
    """
    On the Identifiability of the Post-Nonlinear Causal Model

    References
    ----------
    https://arxiv.org/ftp/arxiv/papers/1205/1205.2599.pdf

    Parameters
    ----------
    hidden_layers: int
        number of hidden layer of mlp
    hidden_units: int
        number of unit of per hidden layer
    batch_size: int
        size of training batch
    epochs: int
        training times on all samples
    lr: float
        learning rate
    alpha: float
        significance level
    bias: bool
        whether use bias
    activation: callable
        nonlinear activation function
    device_type: str
        'cpu' or 'gpu', default: 'cpu'
    device_ids: int or str
        e.g. 0 or '0,1', denotes which gpu that you want to use.

    Examples
    --------
    >>> from castle.algorithms.gradient.pnl.torch import PNL
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> n = PNL()
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self, hidden_layers=1, hidden_units=10, batch_size=64,
                 epochs=100, lr=1e-4, alpha=0.01, bias=True,
                 activation=nn.LeakyReLU(), device_type='cpu', device_ids=None):
        super(PNL, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.bias = bias
        self.activation = activation
        self.device_type = device_type
        self.device_ids = device_ids
        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type='cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

    def learn(self, data, columns=None, **kwargs):

        n_nodes = data.shape[1]
        g = np.zeros((n_nodes, n_nodes))

        all_nodes_pair = itertools.permutations(range(n_nodes), 2)
        for i, j in all_nodes_pair:
            x1 = torch.tensor(data[:, i], device=self.device).unsqueeze(-1)
            x2 = torch.tensor(data[:, j], device=self.device).unsqueeze(-1)

            # initialize model and parameters
            l1 = MLP(input_dim=1, hidden_layers=self.hidden_layers,
                     hidden_units=self.hidden_units, output_dim=1,
                     bias=self.bias, activation=self.activation,
                     device=self.device)
            l2 = MLP(input_dim=1, hidden_layers=self.hidden_layers,
                     hidden_units=self.hidden_units, output_dim=1,
                     bias=self.bias, activation=self.activation,
                     device=self.device)
            optimizer = torch.optim.SGD([{'params': l1.parameters()},
                                         {'params': l2.parameters()}],
                                        lr=self.lr)
            # nonlinear ICA
            e2 = self._nonlinear_ica(l1, l2, x1, x2, optimizer=optimizer)

            # kernel-based independent test
            ind = hsic_test(x1.cpu().detach().numpy(),
                            e2.cpu().detach().numpy(), alpha=self.alpha)
            if ind == 0:  # x1->x2
                g[i, j] = 1

        self.causal_matrix = Tensor(g, index=columns, columns=columns)

    def _nonlinear_ica(self, f1, f2, x1, x2, optimizer):

        batch_generator = batch_loader(x1, x2, batch_size=self.batch_size)
        for i in range(self.epochs):
            for x1_batch, x2_batch in batch_generator:
                optimizer.zero_grad()

                l2_jacob = torch.diag(compute_jacobian(f2, x2_batch).squeeze())
                e2 = f2(x2_batch) - f1(x1_batch)
                entropy = compute_entropy(e2)
                loss = entropy - torch.log(torch.abs(l2_jacob)).sum()

                loss.backward()
                optimizer.step()

        e2 = f2(x2) - f1(x1)

        return e2
