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
import scipy.linalg as slin
import scipy.optimize as sopt
from loguru import logger

from castle.common import BaseLearner, Tensor


class NotearsLowRank(BaseLearner):
    """
    NotearsLowRank Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/2006.05691
    
    Examples
    --------
    >>> from castle.algorithms import NotearsLowRank
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> true_dag, X = load_dataset(name='iid_test')
    >>> rank = np.linalg.matrix_rank(true_dag)
    >>> n = NotearsLowRank()
    >>> n.learn(X, rank=rank)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self):

        super().__init__()

    def learn(self, data, rank):
        """
        Set up and run the NotearsLowRank algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        rank: int
            The rank of data.
        """
        if isinstance(data, np.ndarray):
            X = data
        elif isinstance(data, Tensor):
            X = data.data
        else:
            raise TypeError('The type of data must be '
                            'Tensor or numpy.ndarray, but got {}'
                            .format(type(data)))

        n, d = X.shape
        random_cnt = 0
        total_cnt = 0
        while total_cnt <= 20:
            try:
                if total_cnt == 0:
                    w_init_ = np.zeros((d,d))
                else:
                    w_init_ = np.random.uniform(-0.3, 0.3, (d,d))
                
                causal_matrix = self.notears_low_rank(X, rank, w_init_)
                
                random_cnt += 1
                total_cnt += 1
                if random_cnt >= 1:
                    break

            except ValueError:
                print(total_cnt, 'NAN error')
                total_cnt += 1

        self.causal_matrix = causal_matrix

    def notears_low_rank(self, X, rank, w_init=None, tcnt=0, max_iter=15, 
            h_tol=1e-6, rho_max=1e+20, w_threshold=0.3, save_folder=None):
        """
        Solve min_W ell(W; X) s.t. h(W) = 0 using augmented Lagrangian.

        Parameters
        ----------
        X: [n,d] sample matrix
            max_iter: max number of dual ascent steps.
        rank: int
            The rank of data.
        h_tol: exit if |h(w)| <= h_tol
            w_threshold: fixed threshold for edge weights.

        Return
        ------
        W_est: np.ndarray
            estimate [d,d] dag matrix
        """
        def _h(W):
            return np.trace(slin.expm(W * W)) - d

        def _func(uv):
            # L = 0.5/n * || X (I - UV) ||_F^2 + rho/2*h^2 + alpha*h
            nn = len(uv)
            u = uv[0: nn // 2]
            u = u.reshape((d, -1))
            v = uv[nn // 2:]
            v = v.reshape((d, -1))
            W = np.matmul(u, v.transpose())
            loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))
            h = _h(W)
            return loss + 0.5 * rho * h * h + alpha * h

        def _lik(u, v):
            u = u.reshape((d, -1))
            v = v.reshape((d, -1))
            W = np.matmul(u, v.transpose())
            loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d, d) - W), 'fro'))
            return loss

        def _grad(uv):
            nn = len(uv)
            u = uv[0: nn // 2]
            v = uv[nn // 2:]
            gd = np.zeros(nn)
            gd[0: nn // 2] = _grad_u(u, v)
            gd[nn // 2:] = _grad_v(v, u)
            return gd

        def _grad_u(u, v):
            # -2⋅X⊤⋅(X−X⋅U⋅V⊤)⋅V
            # ( expm(t2) .* 2(u*v') ) * v, t2 = vu' .* vu'
            u = u.reshape((d, -1))
            v = v.reshape((d, -1))
            W = np.matmul(u, v.transpose())
            loss_grad = - 1.0 / n * X.T.dot(X).dot(np.eye(d, d) - W).dot(v)
            E = slin.expm(W * W)  # expm(t2)'
            obj_grad = loss_grad + (rho * (np.trace(E) - d) + alpha) * 2 * \
                       np.matmul(E.T * W, v)
            return obj_grad.flatten()

        def _grad_v(v, u):
            # −2⋅(X⊤−V⋅U⊤⋅X⊤)⋅X⋅U
            # ( expm(t1) .* 2(v*u') ) * u, t1 = uv' .* uv'
            u = u.reshape((d, -1))
            v = v.reshape((d, -1))
            W = np.matmul(v, u.transpose())
            loss_grad = - 1.0 / n * (np.eye(d, d) - W).dot(X.T).dot(X).dot(u)
            E = slin.expm(W * W)  # expm(t1)'
            obj_grad = loss_grad + (rho * (np.trace(E) - d) + alpha) * 2 * \
                       np.matmul(E.T * W, u)
            return obj_grad.flatten()

        n, d = X.shape
        r = rank
        if w_init is None:
            w_init = np.zeros((d,d))

        u, s, vt = np.linalg.svd(w_init)
        u_new = u[:, range(r)].dot(np.diag(s[range(r)])).reshape(d*r)
        v_new = vt[range(r), :].transpose().reshape(d*r)
        
        if np.sum(np.abs(u_new)) <= 1e-6 and np.sum(np.abs(v_new)) <= 1e-6:
            raise ValueError('nearly zero gradient; input new initialized W')
    
        rho, alpha, h, h_new = 1.0, 0.0, np.inf, np.inf
        uv_new = np.hstack((u_new, v_new))
        uv_est = np.copy(uv_new)
        # bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
        
        logger.info('[start]: n={}, d={}, iter_={}, h_={}, rho_={}'.format( \
                    n, d, max_iter, h_tol, rho_max))
        
        for flag in range(-1, max_iter):       
            if flag >= 0: 
                while rho <= rho_max:
                    sol = sopt.minimize(_func, uv_est, method='TNC', 
                                        jac=_grad, options={'disp': False})
                    
                    uv_new = sol.x
                    h_new =_h(np.matmul(uv_new[0: d*r].reshape((d, r)), 
                                        uv_new[d*r:].reshape((d, r)).transpose()))
                    
                    logger.debug(
                        '[iter {}] h={:.3e}, loss={:.3f}, rho={:.1e}'.format( \
                        flag, h_new, _func(uv_new), rho))

                    if h_new > 0.25 * h:
                        rho *= 10
                    else:
                        break
            
            uv_est, h = uv_new, h_new
            
            #############################
            if flag >= 0:
                alpha += rho * h
            
            if flag >= 3 and h <= h_tol:
                break          

        uv_new2 = np.copy(uv_new)
        w_est2 = np.matmul(uv_new2[0: d*r].reshape((d, r)), 
                           uv_new2[d*r:].reshape((d, r)).transpose())
        w_est2 = w_est2.reshape((d, d))
        w_est2[np.abs(w_est2) < w_threshold] = 0
        w_est_binary = np.float32(np.abs(w_est2) >= w_threshold)

        logger.info('FINISHED')

        return w_est_binary
