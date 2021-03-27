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

import math
from copy import deepcopy
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm

from castle.common import BaseLearner, Tensor


def gauss_test(r, n, k, alpha):
    """Gaussian test by Fisher's z-transform for pc algorithms.

    Parameters
    ----------
    r: float
        Correlation coefficient or partial correlation coefficient.
    n: int
        The number of samples.
    k: int
        The number of controlled variables.
    alpha: float
        Significance level.

    Returns
    -------
    out: 0 or 1
        if p >= alpha then reject h0, return 0 else return 1
    """

    cut_at = 0.99999
    r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1
    # Fisher’s z-transform
    res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
    # p-value H0: r == 0 H1: r != 0
    p = 2 * (1 - norm.cdf(abs(res)))

    if p >= alpha:
        return 0
    else:
        return 1


class PC(BaseLearner):
    """PC Algorithm.

    A classic causal discovery algorithm based on conditional independence tests.

    Reference: https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf

    Attributes
    ----------
    n_samples : int
        Number of samples.
    causal_matrix : numpy.ndarray
        Learned causal structure matrix.

    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import load_dataset

    >>> true_dag, X = load_dataset(name='iid_test')
    >>> pc = PC()
    >>> pc.learn(X)
    >>> GraphDAG(pc.causal_matrix, true_dag, 'result_pc')
    >>> met = MetricsDAG(pc.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self):

        super(PC, self).__init__()

    def learn(self, data, test_method=gauss_test, alpha=0.05):
        """Set up and run the PC algorithm.

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            Training data.
        test_method: function
            The approach of conditional independent hypothesis test.
        alpha: float
            Significance level.
        """

        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        elif isinstance(data, Tensor):
            data = pd.DataFrame(data.data)
        else:
            raise TypeError('The type of tensor must be '
                            'Tensor or numpy.ndarray, but got {}'
                            .format(type(data)))

        self.n_samples = data.shape[0]
        # Calculate the correlation coefficient in all variables.
        corr = data.corr()

        # Generating an undirected skeleton matrix
        skeleton, sep_set = self._cal_skeleton(corr_matrix=corr,
                                               test_method=test_method,
                                               alpha=alpha)

        # Generating an causal matrix (DAG)
        self._causal_matrix = self._cal_cpdag(skeleton,
                                              sep_set).astype(int).values

    def _cal_skeleton(self, corr_matrix, test_method, alpha):
        """Generate an undirected skeleton matrix.

        Parameters
        ----------
        corr_matrix: pandas.DataFrame
            Correlation coefficient matrix of all variables.
            Shape is d * d, where d is the number of variables.
        test_method: function
            The approach of conditional independent hypothesis test.
        alpha: float
            Significance level.

        Return
        ------
        skeleton_matrix: pandas.DataFrame
        sep_set: dict, d-separation sets
            The key is two variables '(x, y)' to test,
            the value is set that the elements make x and y conditional
            independent.
        """

        sep_set = {}
        skeleton = deepcopy(corr_matrix)
        variables = list(skeleton.columns)
        for i, label in enumerate(variables):
            skeleton.iloc[i, i] = 0
        k = 0
        # if k == 0, test the correlation coefficient
        # if k == 1, test the first order partial correlation coefficient
        # ...
        ij = list(combinations(corr_matrix.columns, 2))
        while k <= len(variables) - 2:
            for i, j in ij:
                if k == 0:
                    r = corr_matrix.loc[i, j]
                    bool = test_method(r=r, n=self.n_samples,
                                       k=k, alpha=alpha)
                    skeleton.loc[i, j] = skeleton.loc[j, i] = bool
                    if not bool:
                        sep_set[(i, j)] = []
                else:
                    if skeleton.loc[i, j] == 0:
                        continue
                    else:
                        # combined sub_matrix
                        sub_v = deepcopy(variables)
                        sub_v.remove(i)
                        sub_v.remove(j)
                        a = []
                        for ks in combinations(sub_v, k):
                            sub = [i, j]
                            for s in ks:
                                sub.append(s)
                            sub_corr = corr_matrix.loc[sub, sub]
                            # inverse matrix
                            PM = np.linalg.inv(sub_corr)
                            r = -1 * PM[0, 1] / math.sqrt(
                                abs(PM[0, 0] * PM[1, 1]))
                            bool = test_method(r=r, n=self.n_samples,
                                               k=k, alpha=alpha)
                            if not bool:
                                a.extend(ks)
                            if a:
                                skeleton.loc[i, j] = skeleton.loc[j, i] = 0
                                sep_set[(i, j)] = a
                                # break # Don't break if you want to check all ks.
                            else:
                                skeleton.loc[i, j] = skeleton.loc[j, i] = 1
            k += 1
        return skeleton, sep_set

    def _cal_cpdag(self, skeleton_matrix, sep_set):
        """Extend the skeleton to a CPDAG.

        Pre-processing rule:
        for all pairs of nonadjacent variables (i, j) with common neighbour k do
            if k not in sep_set(i, j) then
                Replace i——k——j in skeleton by i——>k<——j

        Parameters
        ----------
        skeleton_matrix: pandas.DataFrame
        sep_set: dict，d-separation sets
            The key is two variables '(x, y)' to test,
            the value is set that the elements make x and y conditional
            independent.

        Returns
        -------
        out: pandas.DataFrame
            A matrix of CPDAG.
        """

        pdag = deepcopy(skeleton_matrix)
        # pre-processing
        for ij in sep_set.keys():
            i, j = ij
            all_k = [x for x in pdag.columns if x not in ij]
            for k in all_k:
                if pdag.loc[i, k] + pdag.loc[k, i] != 0 \
                        and pdag.loc[k, j] + pdag.loc[j, k] != 0:
                    if k not in sep_set[ij]:
                        if pdag.loc[i, k] + pdag.loc[k, i] == 2:
                           pdag.loc[k, i] = 0
                        if pdag.loc[j, k] + pdag.loc[k, j] == 2:
                           pdag.loc[k, j] = 0
        pdag = self._rule_1(pdag=pdag)
        pdag = self._rule_2(pdag=pdag)
        pdag = self._rule_3(pdag=pdag, sep_set=sep_set)
        pdag = self._rule_4(pdag=pdag, sep_set=sep_set)

        return pdag

    def _rule_1(self, pdag):
        """Rule_1

        Orient i——j into i——>j whenever there is an arrow k——>i
        such that k and j are nonadjacent.

        Parameters
        ----------
        pdag: pandas.DataFrame

        Returns
        -------
        out: pandas.DataFrame, pdag
        """

        ind = list(combinations(pdag.columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if pdag.loc[i, j] * pdag.loc[j, i] == 0:
                continue
            # search i——j
            else:
                all_k = [x for x in pdag.columns if x not in ij]
                for k in all_k:
                    if pdag.loc[k, i] == 1 and pdag.loc[i, k] == 0 \
                            and pdag.loc[k, j] + pdag.loc[j, k] == 0:
                        pdag.loc[j, i] = 0
        return pdag

    def _rule_2(self, pdag):
        """Rule_2

        Orient i——j into i——>j whenever there is a chain i——>k——>j.

        Parameters
        ----------
        pdag: pandas.DataFrame

        Returns
        -------
        out: pandas.DataFrame, pdag
        """
        ind = list(combinations(pdag.columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if pdag.loc[i, j] * pdag.loc[j, i] == 0:
                continue
            # search i——j
            else:
                all_k = [x for x in pdag.columns if x not in ij]
                for k in all_k:
                    if pdag.loc[i, k] ==1 and pdag.loc[k, i] == 0 \
                            and pdag.loc[k, j] ==1 \
                            and pdag.loc[j, k] == 0:
                        pdag.loc[j, i] = 0
        return pdag

    def _rule_3(self, pdag, sep_set=None):
        """Rule_3

        Orient i——j into i——>j
        whenever there are two chains i——k——>j and i——l——>j
        such that k and l are non-adjacent.

        Parameters
        ----------
        pdag: pandas.DataFrame
        sep_set: dict，d-separation sets
            The key is two variables '(x, y)' to test,
            the value is set that the elements make x and y conditional
            independent.

        Returns
        -------
        out: pandas.DataFrame, pdag
        """
        ind = list(combinations(pdag.columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if pdag.loc[i, j] * pdag.loc[j, i] == 0:
                continue
            # search i——j
            else:
                for kl in sep_set.keys(): # k and l are nonadjacent.
                    k, l = kl
                    # if i——k——>j and  i——l——>j
                    if pdag.loc[i, k] == 1 \
                            and pdag.loc[k, i] == 1 \
                            and pdag.loc[k, j] == 1 \
                            and pdag.loc[j, k] == 0 \
                            and pdag.loc[i, l] == 1 \
                            and pdag.loc[l, i] == 1 \
                            and pdag.loc[l, j] == 1 \
                            and pdag.loc[j, l] == 0:
                        pdag.loc[j, i] = 0
        return pdag

    def _rule_4(self, pdag, sep_set=None):
        """Rule_4

        Orient i——j into i——>j
        whenever there are two chains i——k——>l and k——>l——>j
        such that k and j are non-adjacent.

        Parameters
        ----------
        pdag: pandas.DataFrame
        sep_set: dict，d-separation sets
            The key is two variables '(x, y)' to test,
            the value is set that the elements make x and y conditional
            independent.

        Returns
        -------
        out: pandas.DataFrame, pdag
        """
        ind = list(combinations(pdag.columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if pdag.loc[i, j] * pdag.loc[j, i] == 0:
                continue
            # search i——j
            else:
                for kj in sep_set.keys():  # k and j are nonadjacent.
                    if j not in kj:
                        continue
                    else:
                        kj = list(kj)
                        kj.remove(j)
                        k = kj[0]
                        ls = [x for x in pdag.columns
                              if x not in [i, j, k]]
                        for l in ls:
                            if pdag.loc[k, l] == 1 \
                                    and pdag.loc[l, k] == 0 \
                                    and pdag.loc[i, k] == 1 \
                                    and pdag.loc[k, i] == 1 \
                                    and pdag.loc[l, j] == 1 \
                                    and pdag.loc[j, l] == 0:
                                pdag.loc[j, i] = 0
        return pdag
