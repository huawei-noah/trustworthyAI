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

from copy import deepcopy
from itertools import combinations
import numpy as np

from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import CI_Test


class PC(BaseLearner):
    """PC Algorithm.

    A classic causal discovery algorithm based on conditional independence tests.

    Reference: https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf

    Parameters
    ----------
    alpha: float, default 0.05
        Significance level.
    ci_test : str
        ci_test method

    Attributes
    ----------
    causal_matrix : array
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

    def __init__(self, alpha=0.05, ci_test='gauss'):

        super(PC, self).__init__()
        self.alpha = alpha
        self.causal_matrix = None
        self.ci_test = ci_test

    def learn(self, data, **kwargs):
        """Set up and run the PC algorithm.

        Parameters
        ----------
        data: array or Tensor
            Training data.
        """

        if isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, Tensor):
            data = data.data
        else:
            raise TypeError('The type of tensor must be '
                            'Tensor or array, but got {}.'
                            .format(type(data)))
        # Generating an undirected skeleton matrix
        skeleton, sep_set = FindSkeleton.origin_pc(data=data, alpha=self.alpha,
                                                   ci_test=self.ci_test)
        # Generating an causal matrix (DAG)
        self._causal_matrix = orient(skeleton, sep_set).astype(int)


class FindSkeleton(object):
    """Contains multiple methods for finding skeleton"""

    @staticmethod
    def origin_pc(data, alpha=0.05, ci_test='gauss'):
        """Origin PC-algorithm for learns a skeleton graph

        It learns a skeleton graph which contains only undirected edges
        from data. This is the original version of the PC-algorithm for the
        skeleton.

        Parameters
        ----------
        data : array, (n_samples, n_features)
            Dataset with a set of variables V
        alpha : float, default 0.05
            significant level
        ci_test : str
            ci_test method

        Returns
        -------
        skeleton : array
            The undirected graph
        seq_set : dict
            Separation sets
            Such as key is (x, y), then value is a set of other variables
            not contains x and y.
        """

        n_features = data.shape[1]
        skeleton = np.ones((n_features, n_features)) - np.eye(n_features)
        nodes = list(range(n_features))
        sep_set = {}
        k = 0
        while k <= n_features - 2:
            for i, j in combinations(nodes, 2):
                if k == 0:
                    if ci_test == 'gauss':
                        p_value = CI_Test.gauss_test(data, i, j, ctrl_var=[])
                    else:
                        raise ValueError('Unknown ci_test method, please check '
                                         f'the parameter {ci_test}.')
                    if p_value >= alpha:
                        skeleton[i, j] = skeleton[j, i] = 0
                        sep_set[(i, j)] = []
                    else:
                        pass
                else:
                    if skeleton[i, j] == 0:
                        continue
                    other_nodes = deepcopy(nodes)
                    other_nodes.remove(i)
                    other_nodes.remove(j)
                    s = []
                    for ctrl_var in combinations(other_nodes, k):
                        ctrl_var = list(ctrl_var)
                        if ci_test == 'gauss':
                            p_value = CI_Test.gauss_test(data, i, j, ctrl_var)
                        else:
                            raise ValueError('Unknown ci_test method, please check '
                                             f'the parameter {ci_test}.')
                        if p_value >= alpha:
                            s.extend(ctrl_var)
                        if s:
                            skeleton[i, j] = skeleton[j, i] = 0
                            sep_set[(i, j)] = s
                            break
            k += 1

        return skeleton, sep_set


def orient(skeleton, sep_set):
    """Extending the Skeleton to the Equivalence Class

    it orients the undirected edges to form an equivalence class of DAGs.

    Parameters
    ----------
    skeleton : array
        The undirected graph
    sep_set : dict
        separation sets
        if key is (x, y), then value is a set of other variables
        not contains x and y

    Returns
    -------
    out : array
        An equivalence class of DAGs can be uniquely described
        by a completed partially directed acyclic graph (CPDAG)
        which includes both directed and undirected edges.
    """

    def _rule_1(cpdag):
        """Rule_1

        Orient i——j into i——>j whenever there is an arrow k——>i
        such that k and j are nonadjacent.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
                continue
            # search i——j
            else:
                all_k = [x for x in columns if x not in ij]
                for k in all_k:
                    if cpdag[k, i] == 1 and cpdag[i, k] == 0 \
                            and cpdag[k, j] + cpdag[j, k] == 0:
                        cpdag[j, i] = 0
        return cpdag

    def _rule_2(cpdag):
        """Rule_2

        Orient i——j into i——>j whenever there is a chain i——>k——>j.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
                continue
            # search i——j
            else:
                all_k = [x for x in columns if x not in ij]
                for k in all_k:
                    if cpdag[i, k] == 1 and cpdag[k, i] == 0 \
                            and cpdag[k, j] == 1 \
                            and cpdag[j, k] == 0:
                        cpdag[j, i] = 0
        return cpdag

    def _rule_3(cpdag, sep_set=None):
        """Rule_3

        Orient i——j into i——>j
        whenever there are two chains i——k——>j and i——l——>j
        such that k and l are non-adjacent.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
                continue
            # search i——j
            else:
                for kl in sep_set.keys():  # k and l are nonadjacent.
                    k, l = kl
                    # if i——k——>j and  i——l——>j
                    if cpdag[i, k] == 1 \
                            and cpdag[k, i] == 1 \
                            and cpdag[k, j] == 1 \
                            and cpdag[j, k] == 0 \
                            and cpdag[i, l] == 1 \
                            and cpdag[l, i] == 1 \
                            and cpdag[l, j] == 1 \
                            and cpdag[j, l] == 0:
                        cpdag[j, i] = 0
        return cpdag

    def _rule_4(cpdag, sep_set=None):
        """Rule_4

        Orient i——j into i——>j
        whenever there are two chains i——k——>l and k——>l——>j
        such that k and j are non-adjacent.
        """

        columns = list(range(cpdag.shape[1]))
        ind = list(combinations(columns, 2))
        for ij in sorted(ind, key=lambda x: (x[1], x[0])):
            # Iteration every (i, j)
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 0:
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
                        ls = [x for x in columns if x not in [i, j, k]]
                        for l in ls:
                            if cpdag[k, l] == 1 \
                                    and cpdag[l, k] == 0 \
                                    and cpdag[i, k] == 1 \
                                    and cpdag[k, i] == 1 \
                                    and cpdag[l, j] == 1 \
                                    and cpdag[j, l] == 0:
                                cpdag[j, i] = 0
        return cpdag

    columns = list(range(skeleton.shape[1]))
    cpdag = deepcopy(skeleton)
    # pre-processing
    for ij in sep_set.keys():
        i, j = ij
        all_k = [x for x in columns if x not in ij]
        for k in all_k:
            if cpdag[i, k] + cpdag[k, i] != 0 \
                    and cpdag[k, j] + cpdag[j, k] != 0:
                if k not in sep_set[ij]:
                    if cpdag[i, k] + cpdag[k, i] == 2:
                        cpdag[k, i] = 0
                    if cpdag[j, k] + cpdag[k, j] == 2:
                        cpdag[k, j] = 0
    cpdag = _rule_1(cpdag=cpdag)
    cpdag = _rule_2(cpdag=cpdag)
    cpdag = _rule_3(cpdag=cpdag, sep_set=sep_set)
    cpdag = _rule_4(cpdag=cpdag, sep_set=sep_set)

    return cpdag

