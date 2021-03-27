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

import numpy as np

from .models import GAEModel
from .trainers import ALTrainer
from .helpers.tf_utils import set_seed

from castle.common import BaseLearner, Tensor


class Parameters(object):
    def __init__(
            self, seed:'int' = 1230, 
            x_dim:'int' = 1, 
            num_encoder_layers:'int' = 1, 
            num_decoder_layers:'int' = 1, 
            hidden_size:'int' = 4, 
            latent_dim:'int' = 1, 
            l1_graph_penalty:'float' = 0.0, 
            use_float64:'bool' = False, 
            learning_rate:'float' = 1e-3, 
            max_iter:'int' = 10, 
            iter_step:'int' = 3000, 
            init_iter:'int' = 3, 
            h_tol:'float' = 1e-8, 
            init_rho:'float' = 1.0, 
            rho_thres:'float' = 1e+30, 
            h_thres:'float' = 0.25, 
            rho_multiply:'float' = 2.0, 
            early_stopping:'bool' = False, 
            early_stopping_thres:'float' = 1.0, 
            graph_thres:'float' = 0.3):
        self.seed=seed
        self.x_dim=x_dim
        self.num_encoder_layers=num_encoder_layers
        self.num_decoder_layers=num_decoder_layers
        self.hidden_size=hidden_size
        self.latent_dim=latent_dim
        self.l1_graph_penalty=l1_graph_penalty
        self.use_float64=use_float64
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.iter_step=iter_step
        self.init_iter=init_iter
        self.h_tol=h_tol
        self.init_rho=init_rho
        self.rho_thres=rho_thres
        self.h_thres=h_thres
        self.rho_multiply=rho_multiply
        self.early_stopping=early_stopping
        self.early_stopping_thres=early_stopping_thres
        self.graph_thres=graph_thres


class GAE(BaseLearner):
    """
    GAE Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/1911.07420

    Examples
    --------
    >>> from castle.algorithms import GAE
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> true_dag, X = load_dataset(name='iid_test')
    >>> n = GAE()
    >>> n.learn(X, num_encoder_layers=2, num_decoder_layers=2, hidden_size=16,
                max_iter=20, h_tol=1e-12, iter_step=300, rho_thres=1e20, rho_multiply=10, 
                graph_thres=0.2, l1_graph_penalty=1.0, init_iter=5, use_float64=True)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self):
        super().__init__()

    def learn(self, data, **kwargs):
        """
        Set up and run the GAE algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        """
        opt = Parameters()
        for k in kwargs:
            opt.__dict__[k] = kwargs[k]
        
        if isinstance(data, np.ndarray):
            X = data
        elif isinstance(data, Tensor):
            X = data.data
        else:
            raise TypeError('The type of data must be '
                            'Tensor or numpy.ndarray, but got {}'
                            .format(type(data)))
        
        opt.n, opt.d = X.shape[:2]
        if X.ndim == 2:
            X = np.reshape(X,(opt.n, opt.d, 1))
        elif X.ndim == 3:
            opt.x_dim = X.shape[2]
            X = X
        
        causal_matrix = self._gae(X, opt)
        self.causal_matrix = causal_matrix

    def _gae(self, X, opt):
        """
        Starting model of GAE.

        Parameters
        ----------
        X: numpy.ndarray
            The numpy.ndarray format data you want to learn.
        opt: dict
            The parameters dict for gae.
        """

        set_seed(opt.seed)

        model = GAEModel(opt.n, opt.d, opt.x_dim, opt.seed, opt.num_encoder_layers, 
                         opt.num_decoder_layers, opt.hidden_size, opt.latent_dim, 
                         opt.l1_graph_penalty, opt.use_float64)

        trainer = ALTrainer(opt.init_rho, opt.rho_thres, opt.h_thres, 
                            opt.rho_multiply, opt.init_iter, opt.learning_rate, 
                            opt.h_tol, opt.early_stopping, 
                            opt.early_stopping_thres)
        
        W_est = trainer.train(model, X, opt.graph_thres, opt.max_iter, 
                              opt.iter_step)
        W_est = W_est / np.max(np.abs(W_est))
        W_est[np.abs(W_est) < opt.graph_thres] = 0

        return (W_est != 0).astype(int)

