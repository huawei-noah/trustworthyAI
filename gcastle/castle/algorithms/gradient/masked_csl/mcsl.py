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


class Parameters(object):
    def __init__(
            # other settings
            self, use_gpu:'bool' = False,
            seed: 'int' = 1230,
            graph_thres: 'float' = 0.5,
            # model parameters
            model_type:'str' = 'MaskedNN',
            num_hidden_layers:'int' = 4,
            hidden_size:'int'=16,
            l1_graph_penalty:'float' = 2e-3,
            use_float64:'bool' = False,
            # training parameters
            learning_rate:'float' = 3e-2,
            max_iter:'int' = 25,
            iter_step:'int' = 1000,
            init_iter:'int' = 2,
            h_tol:'float' = 1e-10,
            init_rho:'float' = 1e-5,
            rho_thres:'float' = 1e14,
            h_thres:'float' = 0.25,
            rho_multiply:'float' = 10,
            temperature:'float' = 0.2
            ):
        self.seed = seed
        self.use_gpu = use_gpu
        self.graph_thres=graph_thres
        self.model_type=model_type
        self.num_hidden_layers=num_hidden_layers
        self.hidden_size=hidden_size
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
        self.temperature=temperature


class MCSL(BaseLearner):
    """
    MCSL Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

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
    >>> n = MCSL()
    >>> n.learn(X, iter_step=1000, rho_thres=1e14, init_rho=1e-5,
                rho_multiply=10, graph_thres=0.5, l1_graph_penalty=2e-3, 
                degree=2, use_float64=False)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self):
        super().__init__()

    def learn(self, data, **kwargs):
        """
        Set up and run the MCSL algorithm.

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
        
        opt.n, opt.d = X.shape
        pns_mask = np.ones([X.shape[1], X.shape[1]])

        causal_matrix = self._mcsl(X, pns_mask, opt)
        self.causal_matrix = causal_matrix

    def _mcsl(self, X, pns_mask, opt):
        """
        Starting model of MCSL.

        Parameters
        ----------
        X: numpy.ndarray
            The numpy.ndarray format data you want to learn.
        pns_mask: numpy.ndarray
            The mask matrix.
        opt: dict
            The parameters dict for mcsl.
        """

        set_seed(opt.seed)

        if opt.model_type == 'MaskedNN':
            Model = MaskedNN
        elif opt.model_type == 'MaskedQuadraticRegression':
            Model = MaskedQuadraticRegression

        model = Model(opt.n, opt.d, pns_mask, opt.num_hidden_layers,
                      opt.hidden_size, opt.l1_graph_penalty, opt.learning_rate, 
                      opt.seed, opt.use_float64, opt.use_gpu)

        trainer = ALTrainer(opt.init_rho, opt.rho_thres, opt.h_thres, 
                            opt.rho_multiply, opt.init_iter, opt.h_tol, 
                            opt.temperature)

        W_logits = trainer.train(model, X, opt.max_iter, opt.iter_step)

        W_est = callback_after_training(W_logits, opt.temperature, opt.graph_thres)

        return W_est
