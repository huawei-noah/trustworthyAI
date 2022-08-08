# coding=utf-8
# 2021.03 modified (1) logging config
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

import torch
import torch.nn as nn


class GolemModel(nn.Module):
    """
    Set up the objective function of GOLEM.

    Hyperparameters:
        (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
        (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
    """

    def __init__(self, n, d, lambda_1, lambda_2, equal_variances,
                 B_init=None, device=None):
        """
        Initialize self.

        Parameters
        ----------
        n: int
            Number of samples.
        d: int
            Number of nodes.
        lambda_1: float
            Coefficient of L1 penalty.
        lambda_2: float
            Coefficient of DAG penalty.
        equal_variances: bool
            Whether to assume equal noise variances
            for likelibood objective. Default: True.
        B_init: numpy.ndarray or None
            [d, d] weighted matrix for initialization. 
            Set to None to disable. Default: None.
        """

        super().__init__()
        self.n = n
        self.d = d
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init
        self.device= device

        if self.B_init is not None:
            self._B = nn.Parameter(torch.tensor(self.B_init,
                                                device=self.device))
        else:
            self._B = nn.Parameter(torch.zeros((self.d, self.d),
                                               device=self.device))

    def forward(self, X):
        """Build tensorflow graph."""
        # Placeholders and variables
        self.X = X
        self.B = self._preprocess(self._B)

        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood()
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = (self.likelihood + self.lambda_1 * self.L1_penalty 
                      + self.lambda_2 * self.h)

    def _preprocess(self, B):
        """
        Set the diagonals of B to zero.

        Parameters
        ----------
        B: tf.Tensor
            [d, d] weighted matrix.

        Return
        ------
        torch.Tensor: [d, d] weighted matrix.
        """

        return (torch.ones(self.d, device=self.device)
                - torch.eye(self.d, device=self.device)).to(self.device) * B

    def _compute_likelihood(self):
        """
        Compute (negative log) likelihood in the linear Gaussian case.

        Return
        ------
        torch.Tensor: Likelihood term (scalar-valued).
        """
        if self.equal_variances:  # Assuming equal noise variances
            return (0.5 * self.d
                    * torch.log(torch.square(torch.linalg.norm(self.X - self.X @ self.B)))
                    - torch.linalg.slogdet(torch.eye(self.d).to(self.device) - self.B)[1])
        else:  # Assuming non-equal noise variances
            return (0.5
                    * torch.sum(torch.log(torch.sum(torch.square(self.X - self.X @ self.B), dim=0)))
                    - torch.linalg.slogdet(torch.eye(self.d).to(self.device) - self.B)[1])

    def _compute_L1_penalty(self):
        """
        Compute L1 penalty.

        Return
        ------
        tf.Tensor: L1 penalty term (scalar-valued).
        """
        return torch.norm(self.B, p=1)

    def _compute_h(self):
        """
        Compute DAG penalty.

        Return
        ------
        torch.Tensor: DAG penalty term (scalar-valued).
        """
        return torch.trace(torch.matrix_exp(self.B * self.B)) - self.d
