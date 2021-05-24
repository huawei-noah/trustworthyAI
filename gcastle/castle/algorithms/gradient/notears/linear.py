# coding=utf-8
# 2021.03 modified (1) notears_linear(def) to Notears(class)
# 2021.03 added    (1) loguru; 
#                  (2) BaseLearner
# 2021.03 deleted  (1) __main__
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
import numpy as np
import scipy.optimize as sopt
from scipy.special import expit as sigmoid

from castle.common import BaseLearner, Tensor


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Notears(BaseLearner):
    """
    Notears Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Parameters
    ----------
    lambda1: float 
        l1 penalty parameter
    loss_type: str 
        l2, logistic, poisson
    max_iter: int 
        max num of dual ascent steps
    h_tol: float 
        exit if |h(w_est)| <= htol
    rho_max: float 
        exit if rho >= rho_max
    w_threshold: float 
        drop edge if |weight| < threshold

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/1803.01422
    
    Examples
    --------
    >>> from castle.algorithms import Notears
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> true_dag, X = load_dataset(name='iid_test')
    >>> n = Notears()
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self, lambda1=0.1, 
                 loss_type='l2', 
                 max_iter=100, 
                 h_tol=1e-8, 
                 rho_max=1e+16, 
                 w_threshold=0.3):

        super().__init__()

        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

    def learn(self, data):
        """
        Set up and run the Notears algorithm.

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

        causal_matrix = self.notears_linear(X, lambda1=self.lambda1, 
                                            loss_type=self.loss_type, 
                                            max_iter=self.max_iter, 
                                            h_tol=self.h_tol, 
                                            rho_max=self.rho_max, 
                                            w_threshold=self.w_threshold)
        self.causal_matrix = causal_matrix

    def notears_linear(self, X, lambda1, loss_type, max_iter, h_tol, 
                       rho_max, w_threshold):
        """
        Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using 
        augmented Lagrangian.

        Parameters
        ----------
        X: np.ndarray 
            n*d sample matrix

        Return
        ------
        W_est: np.ndarray
            d*d estimated DAG
        """
        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / X.shape[0] * (R ** 2).sum()
                G_loss = - 1.0 / X.shape[0] * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / X.shape[0] * (S - X * M).sum()
                G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _h(W):
            """
            Evaluate value and gradient of acyclicity constraint.
            """
            #     E = slin.expm(W * W)  # (Zheng et al. 2018)
            #     h = np.trace(E) - d
            M = np.eye(d) + W * W / d  # (Yu et al. 2019)
            E = np.linalg.matrix_power(M, d - 1)
            h = (E.T * M).sum() - d
            G_h = E.T * W * 2
            return h, G_h

        def _adj(w):
            """
            Convert doubled variables ([2 d^2] array) back to original 
            variables ([d, d] matrix).
            """
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """
            Evaluate value and gradient of augmented Lagrangian for 
            doubled variables ([2 d^2] array).
            """
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), 
                                   axis=None)
            return obj, g_obj

        n, d = X.shape
        # double w_est into (w_pos, w_neg)
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) 
                for i in range(d) for j in range(d)]
        if loss_type == 'l2':
            X = X - np.mean(X, axis=0, keepdims=True)
        
        logging.info('[start]: n={}, d={}, iter_={}, h_={}, rho_={}'.format( \
                    n, d, max_iter, h_tol, rho_max))

        for i in range(max_iter):
            w_new, h_new = None, None
            while rho < rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', 
                                    jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                
                logging.info(
                    '[iter {}] h={:.3e}, loss={:.3f}, rho={:.1e}'.format( \
                    i, h_new, _func(w_est)[0], rho))
                
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h

            if h <= h_tol or rho >= rho_max:
                break

        W_est = _adj(w_est)
        W_est[np.abs(W_est) < w_threshold] = 0

        logging.info('FINISHED')

        return (W_est != 0).astype(int)
