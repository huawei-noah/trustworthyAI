# coding=utf-8
# 2021.03 added    (1) logging; 
#                  (2) BaseLearner;
#                  (3) NotearsMLP, NotearsSob;
# 2021.03 deleted  (1) __main__
# 2021.11 added    (1) NotearsNonlinear
#         deleted  (1) NotearsMLP, NotearsSob, MLPModel, SobolevModel
# Huawei Technologies Co., Ltd. 
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Xun Zheng (https://github.com/xunzheng/notears)
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

import logging
import os
import torch
import torch.nn as nn
import numpy as np

from castle.common import BaseLearner, Tensor
from .models import MLPModel, SobolevModel, squared_loss
from .utils.lbfgsb_scipy import LBFGSBScipy
from castle.common.consts import NONLINEAR_NOTEARS_VALID_PARAMS
from castle.common.validator import check_args_value


np.set_printoptions(precision=3)


class NotearsNonlinear(BaseLearner):
    """
    Notears Nonlinear.
    include notears-mlp and notears-sob.
    A gradient-based algorithm using neural network or Sobolev space modeling for non-linear causal relationships.

    Parameters
    ----------
    lambda1: float 
        l1 penalty parameter
    lambda2: float 
        l2 penalty parameter
    max_iter: int 
        max num of dual ascent steps
    h_tol: float 
        exit if |h(w_est)| <= htol
    rho_max: float 
        exit if rho >= rho_max
    w_threshold: float 
        drop edge if |weight| < threshold
    hidden_layers: Iterrable
        Dimension of per hidden layer, and the last element must be 1 as output dimension.
        At least contains 2 elements. For example: hidden_layers=(5, 10, 1), denotes two hidden
        layer has 5 and 10 dimension and output layer has 1 dimension.
        It is effective when model_type='mlp'.
    expansions: int
        expansions of each variable, it is effective when model_type='sob'.
    bias: bool
        Indicates whether to use weight deviation.
    model_type: str
        The Choice of Two Nonlinear Network Models in a Notears Framework:
        Multilayer perceptrons value is 'mlp', Basis expansions value is 'sob'.
    device_type: str, default: cpu
        ``cpu`` or ``gpu``
    device_ids: int or str, default None
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/1909.13189
    
    Examples
    --------
    >>> from castle.algorithms import NotearsNonlinear
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> n = NotearsNonlinear()
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    @check_args_value(NONLINEAR_NOTEARS_VALID_PARAMS)
    def __init__(self, lambda1: float = 0.01,
                 lambda2: float = 0.01,
                 max_iter: int = 100,
                 h_tol: float = 1e-8,
                 rho_max: float = 1e+16,
                 w_threshold: float = 0.3,
                 hidden_layers: tuple = (10, 1),
                 expansions: int = 10,
                 bias: bool = True,
                 model_type: str = "mlp",
                 device_type: str = "cpu",
                 device_ids=None):

        super().__init__()

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold
        self.hidden_layers = hidden_layers
        self.expansions = expansions
        self.bias = bias
        self.model_type = model_type
        self.device_type = device_type
        self.device_ids = device_ids
        self.rho, self.alpha, self.h = 1.0, 0.0, np.inf

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
        """
        Set up and run the NotearsNonlinear algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """
        X = Tensor(data, columns=columns)

        input_dim = X.shape[1]
        model = self.get_model(input_dim)
        if model:
            W_est = self.notears_nonlinear(model, X)

            causal_matrix = (abs(W_est) > self.w_threshold).astype(int)
            self.weight_causal_matrix = Tensor(W_est,
                                               index=X.columns,
                                               columns=X.columns)
            self.causal_matrix = Tensor(causal_matrix, index=X.columns, columns=X.columns)

    def dual_ascent_step(self, model, X):
        """
        Perform one step of dual ascent in augmented Lagrangian.

        Parameters
        ----------
        model: nn.Module
            network model
        X: torch.tenser
            sample data

        Returns
        -------
        :tuple
            cycle control parameter
        """
        h_new = None
        optimizer = LBFGSBScipy(model.parameters())
        X_torch = torch.from_numpy(X)
        while self.rho < self.rho_max:
            X_torch = X_torch.to(self.device)

            def closure():
                optimizer.zero_grad()
                X_hat = model(X_torch)
                loss = squared_loss(X_hat, X_torch)
                h_val = model.h_func()
                penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
                l2_reg = 0.5 * self.lambda2 * model.l2_reg()
                l1_reg = self.lambda1 * model.fc1_l1_reg()
                primal_obj = loss + penalty + l2_reg + l1_reg
                primal_obj.backward()
                return primal_obj

            optimizer.step(closure, self.device)  # NOTE: updates model in-place
            with torch.no_grad():
                model = model.to(self.device)
                h_new = model.h_func().item()
            if h_new > 0.25 * self.h:
                self.rho *= 10
            else:
                break
        self.alpha += self.rho * h_new
        self.h = h_new

    def notears_nonlinear(self,
                          model: nn.Module,
                          X: np.ndarray):
        """
        notaears frame entrance.

        Parameters
        ----------
        model: nn.Module
            network model
        X: castle.Tensor or numpy.ndarray
            sample data

        Returns
        -------
        :tuple
            Prediction Graph Matrix Coefficients.
        """
        logging.info('[start]: n={}, d={}, iter_={}, h_={}, rho_={}'.format(
            X.shape[0], X.shape[1], self.max_iter, self.h_tol, self.rho_max))

        for _ in range(self.max_iter):
            self.dual_ascent_step(model, X)

            logging.debug('[iter {}] h={:.3e}, rho={:.1e}'.format(_, self.h, self.rho))

            if self.h <= self.h_tol or self.rho >= self.rho_max:
                break
        W_est = model.fc1_to_adj()

        logging.info('FINISHED')

        return W_est

    def get_model(self, input_dim):
        """
            Choose a different model.
        Parameters
        ----------
        input_dim: int
            Enter the number of data dimensions.

        Returns
        -------

        """
        if self.model_type == "mlp":
            model = MLPModel(dims=[input_dim, *self.hidden_layers],
                             bias=self.bias, device=self.device)
            return model
        elif self.model_type == "sob":
            model = SobolevModel(input_dim, k=self.expansions, bias=self.bias,
                                 device=self.device)
            return model
        else:
            logging.info(f'Unsupported model type {self.model_type}.')
