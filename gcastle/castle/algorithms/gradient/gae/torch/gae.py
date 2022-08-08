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

import os
import random
import torch
import logging
import numpy as np

from castle.common import BaseLearner, Tensor
from .trainers.al_trainer import ALTrainer
from .models.model import AutoEncoder


def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except Exception:
        pass


class GAE(BaseLearner):
    """
    GAE Algorithm.
    A gradient-based algorithm using graph autoencoder to model non-linear
    causal relationships.

    Parameters
    ----------
    input_dim: int, default: 1
        dimension of vector for x
    hidden_layers: int, default: 1
        number of hidden layers for encoder and decoder
    hidden_dim: int, default: 4
        hidden size for mlp layer
    activation: callable, default: nn.LeakyReLU(0.05)
        nonlinear functional
    epochs: int, default: 10
        Number of iterations for optimization problem
    update_freq: int, default: 3000
        Number of steps for each iteration
    init_iter: int, default: 3
        Initial iteration to disallow early stopping
    lr: float, default: 1e-3
        learning rate
    alpha: float, default: 0.0
        Lagrange multiplier
    beta: float, default: 2.0
        Multiplication to amplify rho each time
    init_rho: float, default: 1.0
        Initial value for rho
    rho_thresh: float, default: 1e30
        Threshold for rho
    gamma: float, default: 0.25
        Threshold for h
    penalty_lambda: float, default: 0.0
        L1 penalty for sparse graph. Set to 0.0 to disable
    h_thresh: float, default: 1e-8
        Tolerance of optimization problem
    graph_thresh: float, default: 0.3
        Threshold to filter out small values in graph
    early_stopping: bool, default: False
        Whether to use early stopping
    early_stopping_thresh: float, default: 1.0
        Threshold ratio for early stopping
    seed: int, default: 1230
        Reproducibility, must be int
    device_type: str, default: 'cpu'
        'cpu' or 'gpu'
    device_ids: int or str, default '0'
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str,
        e.g. 0 or '0', For multi-device modules, ``device_ids`` must be str,
        format like '0, 1'.
    """

    def __init__(self,
                 input_dim=1,
                 hidden_layers=1,
                 hidden_dim=4,
                 activation=torch.nn.LeakyReLU(0.05),
                 epochs=10,
                 update_freq=3000,
                 init_iter=3,
                 lr=1e-3,
                 alpha=0.0,
                 beta=2.0,
                 init_rho=1.0,
                 rho_thresh=1e30,
                 gamma=0.25,
                 penalty_lambda=0.0,
                 h_thresh=1e-8,
                 graph_thresh=0.3,
                 early_stopping=False,
                 early_stopping_thresh=1.0,
                 seed=1230,
                 device_type='cpu',
                 device_ids='0'):

        super(GAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.epochs = epochs
        self.update_freq = update_freq
        self.init_iter = init_iter
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.init_rho = init_rho
        self.rho_thresh = rho_thresh
        self.gamma = gamma
        self.penalty_lambda = penalty_lambda
        self.h_thresh = h_thresh
        self.graph_thresh = graph_thresh
        self.early_stopping = early_stopping
        self.early_stopping_thresh = early_stopping_thresh
        self.seed = seed
        self.device_type = device_type
        self.device_ids = device_ids

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

    def learn(self, data, columns=None, **kwargs):

        x = torch.from_numpy(data)

        self.n, self.d = x.shape[:2]
        if x.ndim == 2:
            x = x.reshape((self.n, self.d, 1))
            self.input_dim = 1
        elif x.ndim == 3:
            self.input_dim = x.shape[2]

        w_est = self._gae(x).detach().cpu().numpy()

        self.weight_causal_matrix = Tensor(w_est,
                                           index=columns,
                                           columns=columns)
        causal_matrix = (abs(w_est) > self.graph_thresh).astype(int)
        self.causal_matrix = Tensor(causal_matrix,
                                    index=columns,
                                    columns=columns)

    def _gae(self, x):

        set_seed(self.seed)
        model = AutoEncoder(d=self.d,
                            input_dim=self.input_dim,
                            hidden_layers=self.hidden_layers,
                            hidden_dim=self.hidden_dim,
                            activation=self.activation,
                            device=self.device,
                            )
        trainer = ALTrainer(n=self.n,
                            d=self.d,
                            model=model,
                            lr=self.lr,
                            init_iter=self.init_iter,
                            alpha=self.alpha,
                            beta=self.beta,
                            rho=self.init_rho,
                            l1_penalty=self.penalty_lambda,
                            rho_thresh=self.rho_thresh,
                            h_thresh=self.h_thresh,  # 1e-8
                            early_stopping=self.early_stopping,
                            early_stopping_thresh=self.early_stopping_thresh,
                            gamma=self.gamma,
                            seed=self.seed,
                            device=self.device)
        w_est = trainer.train(x=x,
                              epochs=self.epochs,
                              update_freq=self.update_freq)
        w_est = w_est / torch.max(abs(w_est))

        return w_est
