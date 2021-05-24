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
import numpy as np

from .helpers.tf_utils import set_seed
from .models.masked_nn import MaskedNN
from .models.masked_quadratic_regression import MaskedQuadraticRegression
from .trainers import ALTrainer
from .helpers.train_utils import callback_after_training

from castle.common import BaseLearner, Tensor

# For logging of tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MCSL(BaseLearner):
    """
    MCSL Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Parameters
    ----------
    use_gpu: bool
        Whether or not to use GPU
    seed: int
        Reproducibility
    graph_thres: float
        Threshold to filter out small values in graph
    model_type: str
        Model type to use [MaskedNN, MaskedQuadraticRegression]
    num_hidden_layers: int
        Number of hidden layers for NN
    hidden_size: int
        Hidden size for NN layers
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
    temperature: float
        Temperature for gumbel sigmoid

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

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
                 rho_multiply=10, graph_thres=0.5, l1_graph_penalty=2e-3, 
                 use_float64=False)
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self, use_gpu=False, seed=1230, graph_thres=0.5, 
                 model_type='MaskedNN', num_hidden_layers=4, hidden_size=16, 
                 l1_graph_penalty=2e-3, use_float64=False, learning_rate=3e-2, 
                 max_iter=25, iter_step=1000, init_iter=2, h_tol=1e-10, 
                 init_rho=1e-5, rho_thres=1e14, h_thres=0.25, rho_multiply=10,
                 temperature=0.2):
        
        super().__init__()
        
        self.use_gpu = use_gpu
        self.seed = seed
        self.graph_thres = graph_thres
        self.model_type = model_type
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
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
        self.temperature = temperature

    def learn(self, data):
        """
        Set up and run the MCSL algorithm.

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
        
        self.n, self.d = X.shape
        pns_mask = np.ones([X.shape[1], X.shape[1]])

        causal_matrix = self._mcsl(X, pns_mask)
        self.causal_matrix = causal_matrix

    def _mcsl(self, X, pns_mask):
        """
        Starting model of MCSL.

        Parameters
        ----------
        X: numpy.ndarray
            The numpy.ndarray format data you want to learn.
        pns_mask: numpy.ndarray
            The mask matrix.
        """

        set_seed(self.seed)

        if self.model_type == 'MaskedNN':
            Model = MaskedNN
        elif self.model_type == 'MaskedQuadraticRegression':
            Model = MaskedQuadraticRegression

        model = Model(self.n, self.d, pns_mask, self.num_hidden_layers,
                      self.hidden_size, self.l1_graph_penalty, self.learning_rate, 
                      self.seed, self.use_float64, self.use_gpu)

        trainer = ALTrainer(self.init_rho, self.rho_thres, self.h_thres, 
                            self.rho_multiply, self.init_iter, self.h_tol, 
                            self.temperature)

        W_logits = trainer.train(model, X, self.max_iter, self.iter_step)

        W_est = callback_after_training(W_logits, self.temperature, self.graph_thres)

        return W_est
