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


class GAE(BaseLearner):
    """
    GAE Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Parameters
    ----------
    seed : int
        Reproducibility, must be int
    x_dim: int
        Dimension of vector for X
    num_encoder_layers: int
        Number of hidden layers for encoder
    num_decoder_layers: int
        Number of hidden layers for decoder
    hidden_size: int
        Hidden size for NN layers
    latent_dim: int
        Latent dimension for autoencoder
    l1_graph_penalty: float
        L1 penalty for sparse graph. Set to 0 to disable
    use_float64: bool
        Whether to use tf.float64 or tf.float32 during training
    learning_rate: float
        Learning rate
    max_iter: int
        Number of iterations for optimization problem
    iter_step: int
        Number of steps for each iteration
    init_iter: int
        Initial iteration to disallow early stopping
    h_tol: float
        Tolerance of optimization problem
    init_rho: float
        Initial value for rho
    rho_thres: float
        Threshold for rho
    h_thres: float
        Threshold for h
    rho_multiply: float
        Multiplication to amplify rho each time
    early_stopping: bool
        Whether to use early stopping
    early_stopping_thres: float
        Threshold ratio for early stopping
    graph_thres: float
        Threshold to filter out small values in graph

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
    >>> n = GAE(num_encoder_layers=2, num_decoder_layers=2, hidden_size=16,
                max_iter=20, h_tol=1e-12, iter_step=300, rho_thres=1e20, rho_multiply=10, 
                graph_thres=0.2, l1_graph_penalty=1.0, init_iter=5, use_float64=True)
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self, seed:'int' = 1230, 
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

        super().__init__()

        self.seed = seed
        self.x_dim = x_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.l1_graph_penalty = l1_graph_penalty
        self.use_float64 = use_float64
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.iter_step = iter_step
        self.init_iter = init_iter
        self.h_tol = h_tol
        self.init_rho = init_rho
        self.rho_thres = rho_thres
        self.h_thres = h_thres
        self.rho_multiply = rho_multiply
        self.early_stopping = early_stopping
        self.early_stopping_thres = early_stopping_thres
        self.graph_thres = graph_thres

    def learn(self, data):
        """
        Set up and run the GAE algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        """

        if isinstance(data, np.ndarray):
            X = data
        elif isinstance(data, Tensor):
            X = data.data
        else:
            raise TypeError('The type of data must be '
                            'Tensor or numpy.ndarray, but got {}'
                            .format(type(data)))
        
        self.n, self.d = X.shape[:2]
        if X.ndim == 2:
            X = np.reshape(X,(self.n, self.d, 1))
        elif X.ndim == 3:
            self.x_dim = X.shape[2]
            X = X
        
        causal_matrix = self._gae(X)
        self.causal_matrix = causal_matrix

    def _gae(self, X):
        """
        Starting model of GAE.

        Parameters
        ----------
        X: numpy.ndarray
            The numpy.ndarray format data you want to learn.
        """

        set_seed(self.seed)

        model = GAEModel(self.n, self.d, self.x_dim, self.seed, self.num_encoder_layers, 
                         self.num_decoder_layers, self.hidden_size, self.latent_dim, 
                         self.l1_graph_penalty, self.use_float64)

        trainer = ALTrainer(self.init_rho, self.rho_thres, self.h_thres, 
                            self.rho_multiply, self.init_iter, self.learning_rate, 
                            self.h_tol, self.early_stopping, 
                            self.early_stopping_thres)
        
        W_est = trainer.train(model, X, self.graph_thres, self.max_iter, 
                              self.iter_step)
        W_est = W_est / np.max(np.abs(W_est))
        W_est[np.abs(W_est) < self.graph_thres] = 0

        return (W_est != 0).astype(int)

