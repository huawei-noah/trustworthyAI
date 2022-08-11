# coding=utf-8
# 2021.03 modified (1) golem(def) to GOLEM(class)
#                  (2) replace tensorflow with pytorch
# 2021.03 added    (1) get_args, set_seed; 
#                  (2) BaseLearner
# 2021.03 deleted  (1) __main__
# Huawei Technologies Co., Ltd. 
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Ignavier Ng (https://github.com/ignavier/golem)
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
import torch
import os

from .golem_utils import GolemModel
from .golem_utils.train import postprocess
from .golem_utils.utils import set_seed

from castle.common import BaseLearner, Tensor


class GOLEM(BaseLearner):
    """
    GOLEM Algorithm.
    A more efficient version of NOTEARS that can reduce number of optimization iterations.

    Paramaters
    ----------
    B_init: None
        File of weighted matrix for initialization. Set to None to disable.
    lambda_1: float
        Coefficient of L1 penalty.
    lambda_2: float
        Coefficient of DAG penalty.
    equal_variances: bool
        Assume equal noise variances for likelibood objective.
    non_equal_variances: bool
        Assume non-equal noise variances for likelibood objective.
    learning_rate: float
        Learning rate of Adam optimizer.
    num_iter: float
        Number of iterations for training.
    checkpoint_iter: int
        Number of iterations between each checkpoint. Set to None to disable.
    seed: int
        Random seed.
    graph_thres: float
        Threshold for weighted matrix.
    device_type: bool
        whether to use GPU or not
    device_ids: int
        choose which gpu to use

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/2006.10201
    
    Examples
    --------
    >>> from castle.algorithms import GOLEM
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> X, true_dag, topology_matrix = load_dataset(name='IID_Test')
    >>> n = GOLEM()
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self, B_init=None,
                 lambda_1=2e-2,
                 lambda_2=5.0,
                 equal_variances=True,
                 non_equal_variances=True,
                 learning_rate=1e-3,
                 num_iter=1e+5,
                 checkpoint_iter=5000,
                 seed=1,
                 graph_thres=0.3,
                 device_type='cpu',
                 device_ids=0):

        super().__init__()

        self.B_init = B_init
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.non_equal_variances = non_equal_variances
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.checkpoint_iter = checkpoint_iter
        self.seed = seed
        self.graph_thres = graph_thres
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
        """
        Set up and run the GOLEM algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        X: numpy.ndarray
            [n, d] data matrix.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        lambda_1: float
            Coefficient of L1 penalty.
        lambda_2: float
            Coefficient of DAG penalty.
        equal_variances: bool
            Whether to assume equal noise variances
            for likelibood objective. Default: True.
        num_iter:int
            Number of iterations for training.
        learning_rate: float
            Learning rate of Adam optimizer. Default: 1e-3.
        seed: int
            Random seed. Default: 1.
        checkpoint_iter: int
            Number of iterations between each checkpoint.
            Set to None to disable. Default: None.
        B_init: numpy.ndarray or None
            [d, d] weighted matrix for initialization.
            Set to None to disable. Default: None.
        """

        X = Tensor(data, columns=columns)
        
        causal_matrix = self._golem(X)
        self.causal_matrix = Tensor(causal_matrix, index=X.columns,
                                    columns=X.columns)

    def _golem(self, X):
        """
        Solve the unconstrained optimization problem of GOLEM, which involves
        GolemModel and GolemTrainer.

        Parameters
        ----------
        X: numpy.ndarray
            [n, d] data matrix.
        
        Return
        ------
        B_result: np.ndarray
            [d, d] estimated weighted matrix.
        
        Hyperparameters
        ---------------
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
        """
        set_seed(self.seed)
        n, d = X.shape
        X = torch.Tensor(X).to(self.device)

        # Set up model
        model = GolemModel(n=n, d=d, lambda_1=self.lambda_1,
                           lambda_2=self.lambda_2,
                           equal_variances=self.equal_variances,
                           B_init=self.B_init,
                           device=self.device)

        self.train_op = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        logging.info("Started training for {} iterations.".format(int(self.num_iter)))
        for i in range(0, int(self.num_iter) + 1):
            model(X)
            score, likelihood, h, B_est = model.score, model.likelihood, model.h, model.B
            
            if i > 0:  # Do not train here, only perform evaluation
                # Optimizer
                self.loss = score
                self.train_op.zero_grad()
                self.loss.backward()
                self.train_op.step()

            if self.checkpoint_iter is not None and i % self.checkpoint_iter == 0:
                logging.info("[Iter {}] score={:.3f}, likelihood={:.3f}, h={:.1e}".format( \
                    i, score, likelihood, h))

        # Post-process estimated solution and compute results
        B_processed = postprocess(B_est.cpu().detach().numpy(), graph_thres=0.3)
        B_result = (B_processed != 0).astype(int)

        return B_result
