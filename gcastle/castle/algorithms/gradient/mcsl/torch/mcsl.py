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
import logging
import random
import numpy as np
import torch

from castle.common.base import BaseLearner, Tensor
from .trainers.al_trainer import Trainer
from .models.masked_model import MaskedModel
from .helpers.utils import callback_after_training
from castle.common.consts import MCSL_VALID_PARAMS
from castle.common.validator import check_args_value


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
    finally:
        pass


class MCSL(BaseLearner):
    """
    Masked Gradient-Based Causal Structure Learning

    A gradient-based algorithm for non-linear additive noise data by learning
    the binary adjacency matrix.

    Parameters
    ----------
    model_type: str, default: 'nn'
        `nn` denotes neural network, `qr` denotes quatratic regression.
    num_hidden_layers: int, default: 4
        Number of hidden layer in neural network when `model_type` is 'nn'.
    hidden_dim: int, default: 16
        Number of hidden dimension in hidden layer, when `model_type` is 'nn'.
    graph_thresh: float, default: 0.5
        Threshold used to determine whether has edge in graph, element greater
        than the `graph_thresh` means has a directed edge, otherwise has not.
    l1_graph_penalty: float, default: 2e-3
        Penalty weight for L1 normalization
    learning_rate: float, default: 3e-2
        learning rate for opitimizer
    max_iter: int, default: 25
        Number of iterations for optimization problem
    iter_step: int, default: 1000
        Number of steps for each iteration
    init_iter: int, default: 2
        Initial iteration to disallow early stopping
    h_tol: float, default: 1e-10
        Tolerance of optimization problem
    init_rho: float, default: 1e-5
        Initial value for penalty parameter.
    rho_thresh: float, default: 1e14
        Threshold for penalty parameter.
    h_thresh: float, default: 0.25
        Threshold for h
    rho_multiply: float, default: 10.0
        Multiplication to amplify rho each time
    temperature: float, default: 0.2
        Temperature for gumbel sigmoid
    device_type: str, default: 'cpu'
        'cpu' or 'gpu'
    device_ids: int or str, default '0'
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.
    random_seed: int, default: 1230
        random seed for every random value

    References
    ----------
    https://arxiv.org/abs/1910.08527

    Examples
    --------
    >>> from castle.algorithms import MCSL
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> true_dag, X = load_dataset(name='iid_test')
    >>> n = MCSL(iter_step=1000, rho_thres=1e14, init_rho=1e-5,
    ...          rho_multiply=10, graph_thres=0.5, l1_graph_penalty=2e-3)
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    @check_args_value(MCSL_VALID_PARAMS)
    def __init__(self, model_type='nn', num_hidden_layers=4, hidden_dim=16,
                 graph_thresh=0.5, l1_graph_penalty=2e-3, learning_rate=3e-2,
                 max_iter=25, iter_step=1000, init_iter=2, h_tol=1e-10,
                 init_rho=1e-5, rho_thresh=1e14, h_thresh=0.25,
                 rho_multiply=10, temperature=0.2, device_type='cpu',
                 device_ids='0', random_seed=1230) -> None:
        super(MCSL, self).__init__()

        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.graph_thresh = graph_thresh
        self.l1_graph_penalty = l1_graph_penalty
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.iter_step = iter_step
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.init_rho = init_rho
        self.rho_thresh = rho_thresh
        self.h_thresh = h_thresh
        self.rho_multiply = rho_multiply
        self.temperature = temperature
        self.device_type = device_type
        self.device_ids = device_ids
        self.random_seed = random_seed

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

    def learn(self, data, columns=None, pns_mask=None, **kwargs) -> None:
        """
        Set up and run the MCSL algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns: Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        pns_mask: array_like or None
            The mask matrix.
            array with element in {0, 1}, ``0`` denotes has no edge in i -> j,
            ``1`` denotes maybe has edge in i -> j or not.
        """

        x = Tensor(data, columns=columns)

        self.n_samples, self.n_nodes = x.shape
        if pns_mask is None:
            pns_mask = torch.ones([x.shape[1], x.shape[1]], device=self.device)
        else:
            pns_mask = torch.tensor(pns_mask, device=self.device)

        causal_matrix, causal_matrix_weight = self._mcsl(x, pns_mask)

        self.causal_matrix_weight = Tensor(causal_matrix_weight,
                                           index=x.columns,
                                           columns=x.columns
                                           )
        self.causal_matrix = Tensor(causal_matrix,
                                    index=x.columns,
                                    columns=x.columns
                                    )

    def _mcsl(self, x, pns_mask) -> tuple:
        """
        Starting model of MCSL.

        Parameters
        ----------
        x: torch.Tensor
            The torch.Tensor data you want to learn.
        pns_mask: torch.Tensor
            The mask matrix.
        """

        set_seed(self.random_seed)

        model = MaskedModel(model_type=self.model_type,
                            n_samples=self.n_samples,
                            n_nodes=self.n_nodes,
                            pns_mask=pns_mask,
                            num_hidden_layers=self.num_hidden_layers,
                            hidden_dim=self.hidden_dim,
                            l1_graph_penalty=self.l1_graph_penalty,
                            seed=self.random_seed,
                            device=self.device)
        trainer = Trainer(model=model,
                          learning_rate=self.learning_rate,
                          init_rho=self.init_rho,
                          rho_thresh=self.rho_thresh,
                          h_thresh=self.h_thresh,
                          rho_multiply=self.rho_multiply,
                          init_iter=self.init_iter,
                          h_tol=self.h_tol,
                          temperature=self.temperature,
                          device=self.device)

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device)
        w_logits = trainer.train(x, self.max_iter, self.iter_step)

        w_est, w_est_weight = callback_after_training(w_logits,
                                                      self.temperature,
                                                      self.graph_thresh)
        return w_est.detach().cpu().numpy(), w_est_weight.detach().cpu().numpy()
