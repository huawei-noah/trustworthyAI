# coding=utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

from abc import ABCMeta, abstractmethod
import math
import numpy as np

from ..functional.utils import cartesian_combination


class DecomposableScore(metaclass=ABCMeta):
    """A base abstract class of scoring criterion"""

    def __init__(self, data):
        self.data = data
        self.n, self.d = data.shape

    @abstractmethod
    def local_score(self, y, pa_y):

        raise NotImplementedError


class BICScore(DecomposableScore):
    """
    Compute local score based on BIC

    Parameters
    ----------
    data: np.ndarray
        sample dataset
    method: str
        one of ['scatter', 'r2']
    """

    def __init__(self, data, method='scatter'):
        super(BICScore, self).__init__(data=data)
        self.method = method
        self._distance = data - np.mean(data, axis=0)

    def local_score(self, y, pa_y):

        if self.method == 'r2':
            return self._bic_by_r2(y, pa_y)
        elif self.method == 'scatter':
            return self._bic_by_scatter(y, pa_y)
        else:
            raise ValueError(f"The parameter `method` must be one of ['r2', 'scatter'], "
                             f"but got {self.method}.")

    def _bic_by_r2(self, y, pa_y):

        Y = self.data[:, [y]]
        k = len(pa_y)
        if len(pa_y) == 0:
            ssr = self.n * np.var(Y, ddof=0)
        else:
            pa = list(pa_y)
            X = self.data[:, pa]
            ssr = np.linalg.lstsq(X, Y, rcond=None)[1]
        tss = np.square(self._distance[y]).sum()

        # cal 1 - R^2_adjust
        adj_r2 = (ssr / (self.n - k - 1)) / (tss / (self.n - 1))
        bic_score = (self.n * np.log(adj_r2) + (k + 1) * np.log(self.n))

        return bic_score

    def _bic_by_scatter(self, y, pa_y):

        scatter = np.cov(self.data, rowvar=False, ddof=0)
        sigma = scatter[y, y]
        pa = list(pa_y)
        k = len(pa)
        if k > 0:
            pa_cov = scatter[pa, :][:, pa]
            y_cov = scatter[y, pa]
            coef = np.linalg.solve(pa_cov, y_cov)
            sigma = sigma - y_cov @ coef
        bic_score = - (self.n * (1 + np.log(sigma)) + (k + 1) * np.log(self.n))

        return bic_score


class BDeuScore(DecomposableScore):
    """
    Bayesian BDeu scoring criterion for discrete variables

    Parameters
    ----------
    data: np.ndarray
        sample dataset
    k: float, default: 0.001
        structure prior
    N: int, default: 10
        prior equivalent sample size
    """

    def __init__(self, data, k=0.001, N=10):
        super(BDeuScore, self).__init__(data=data)
        self.k = k
        self.N = N
        self.r_i_map = {
            i: len(np.unique(self.data[:, i])) for i in range(self.d)
        }

    def local_score(self, y, pa_y):

        # configurations of the parent set pa_y
        arr = [np.unique(self.data[i]) for i in pa_y]
        conf_pa_y = cartesian_combination(arr)

        r_i = self.r_i_map[y]
        q_i = self._q_i(pa_y)

        term0 = (r_i - 1) * q_i * np.log(self.k)
        term1 = 0
        for j_conf in conf_pa_y:
            n_ij = self._cal_nijk(y, pa_y, j_conf)
            term2 = math.lgamma(self.N / q_i) - math.lgamma(n_ij + self.N / q_i)
            states_i = np.unique(self.data[y])
            term3 = 0
            for k in states_i:
                n_ijk = self._cal_nijk(y, pa_y, j_conf, k=k)
                term3 += (math.lgamma(n_ijk + self.N / (r_i * q_i))
                          - math.lgamma(self.N / (r_i * q_i)))
            term1 += (term2 + term3)
        score = term0 + term1

        return score

    def _cal_nijk(self, i, pa_i, j_conf, k=None):
        """
        Nijk is the number of records in D for which X_i = k and
        Pa_i is in the jth configuration.
        """

        pa = list(pa_i)
        if len(pa) != len(j_conf):
            raise ValueError(f"The length of pa_i must be equal to j_conf.")
        if k is None:
            n_ijk = np.prod((self.data[:, pa] == np.array(j_conf)) * 1,
                           axis=1).sum()
        else:
            pa.append(i)
            kj_conf = j_conf.copy()
            kj_conf.append(k)
            n_ijk = np.prod((self.data[:, pa] == np.array(kj_conf)) * 1,
                            axis=1).sum()

        return n_ijk

    def _q_i(self, pa):

        q_i = 1
        for i in pa:
            q_i *= self.r_i_map[i]

        return q_i
