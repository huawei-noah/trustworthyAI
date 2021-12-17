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
from itertools import combinations, permutations
import numpy as np
import joblib

from castle.common import BaseLearner, Tensor
from castle.common.independence_tests import CITest


def _loop(G, d):
    """
    Check if |adj(x, G)\{y}| < d for every pair of adjacency vertices in G

    Parameters
    ----------
    G: numpy.ndarray
        The undirected graph  G
    d: int
        depth of conditional vertices

    Returns
    -------
    out: bool
        if False, denote |adj(i, G)\{j}| < d for every pair of adjacency
        vertices in G, then finished loop.
    """

    assert G.shape[0] == G.shape[1]

    pairs = [(x, y) for x, y in combinations(set(range(G.shape[0])), 2)]
    less_d = 0
    for i, j in pairs:
        adj_i = set(np.argwhere(G[i] == 1).reshape(-1, ))
        z = adj_i - {j}  # adj(C, i)\{j}
        if len(z) < d:
            less_d += 1
    if less_d == len(pairs):
        return False
    else:
        return True


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
    while True:
        old_cpdag = deepcopy(cpdag)
        pairs = list(combinations(columns, 2))
        for ij in pairs:
            i, j = ij
            if cpdag[i, j] * cpdag[j, i] == 1:
                # rule1
                for i, j in permutations(ij, 2):
                    all_k = [x for x in columns if x not in ij]
                    for k in all_k:
                        if cpdag[k, i] == 1 and cpdag[i, k] == 0 \
                                and cpdag[k, j] + cpdag[j, k] == 0:
                            cpdag[j, i] = 0
                # rule2
                for i, j in permutations(ij, 2):
                    all_k = [x for x in columns if x not in ij]
                    for k in all_k:
                        if (cpdag[i, k] == 1 and cpdag[k, i] == 0) \
                            and (cpdag[k, j] == 1 and cpdag[j, k] == 0):
                            cpdag[j, i] = 0
                # rule3
                for i, j in permutations(ij, 2):
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
                # rule4
                for i, j in permutations(ij, 2):
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
        if (cpdag == old_cpdag).all():
            break

    return cpdag


def find_skeleton(data, alpha, ci_test, variant='original', base_skeleton=None,
                  p_cores=1, s=None, batch=None):
    """Find skeleton graph from G using PC algorithm

    It learns a skeleton graph which contains only undirected edges
    from data.

    Parameters
    ----------
    data : array, (n_samples, n_features)
        Dataset with a set of variables V
    alpha : float, default 0.05
        significant level
    ci_test : str, callable
        ci_test method, if str, must be one of [`gauss`, `g2`, `chi2`].
        if callable, must return a tuple that  the last element is `p_value` ,
        like (_, _, p_value) or (chi2, dof, p_value).
        See more: `castle.common.independence_tests.CITest`
    variant : str, default 'original'
        variant of PC algorithm, contains [`original`, `stable`, `parallel`].
        If variant == 'parallel', need to provide the flowing 3 parameters.
    base_skeleton : array, (n_features, n_features)
        prior matrix, must be undirected graph.
        The two conditionals `base_skeleton[i, j] == base_skeleton[j, i]`
        and `and base_skeleton[i, i] == 0` must be satisified which i != j.
    p_cores : int
        Number of CPU cores to be used
    s : bool, default False
        memory-efficient indicator
    batch : int
        number of edges per batch

    if s is None or False, or without batch, batch=|J|.
    |J| denote number of all pairs of adjacency vertices (X, Y) in G.

    Returns
    -------
    skeleton : array
        The undirected graph
    seq_set : dict
        Separation sets
        Such as key is (x, y), then value is a set of other variables
        not contains x and y.

    Examples
    --------
    >>> from castle.algorithms.pc.pc import find_skeleton
    >>> from castle.datasets import load_dataset

    >>> true_dag, X = load_dataset(name='iid_test')
    >>> skeleton, sep_set = find_skeleton(data, 0.05, 'gauss')
    >>> print(skeleton)
    [[0. 0. 1. 0. 0. 1. 0. 0. 0. 0.]
     [0. 0. 0. 1. 1. 1. 1. 0. 1. 0.]
     [1. 0. 0. 0. 1. 0. 0. 1. 0. 0.]
     [0. 1. 0. 0. 1. 0. 0. 1. 0. 1.]
     [0. 1. 1. 1. 0. 0. 0. 0. 0. 1.]
     [1. 1. 0. 0. 0. 0. 0. 1. 1. 1.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 1. 1. 0. 1. 0. 0. 0. 1.]
     [0. 1. 0. 0. 0. 1. 0. 0. 0. 1.]
     [0. 0. 0. 1. 1. 1. 0. 1. 1. 0.]]
    """

    def test(x, y):

        K_x_y = 1
        sub_z = None
        # On X's neighbours
        adj_x = set(np.argwhere(skeleton[x] == 1).reshape(-1, ))
        z_x = adj_x - {y}  # adj(X, G)\{Y}
        if len(z_x) >= d:
            # |adj(X, G)\{Y}| >= d
            for sub_z in combinations(z_x, d):
                sub_z = list(sub_z)
                _, _, p_value = ci_test(data, x, y, sub_z)
                if p_value >= alpha:
                    K_x_y = 0
                    # sep_set[(x, y)] = sub_z
                    break
            if K_x_y == 0:
                return K_x_y, sub_z

        return K_x_y, sub_z

    def parallel_cell(x, y):

        # On X's neighbours
        K_x_y, sub_z = test(x, y)
        if K_x_y == 1:
            # On Y's neighbours
            K_x_y, sub_z = test(y, x)

        return (x, y), K_x_y, sub_z

    if ci_test == 'gauss':
        ci_test = CITest.gauss
    elif ci_test == 'g2':
        ci_test = CITest.g2_test
    elif ci_test == 'chi2':
        ci_test = CITest.chi2_test
    elif callable(ci_test):
        ci_test = ci_test
    else:
        raise ValueError(f'The type of param `ci_test` expect callable,'
                         f'but got {type(ci_test)}.')

    n_feature = data.shape[1]
    if base_skeleton is None:
        skeleton = np.ones((n_feature, n_feature)) - np.eye(n_feature)
    else:
        row, col = np.diag_indices_from(base_skeleton)
        base_skeleton[row, col] = 0
        skeleton = base_skeleton
    nodes = set(range(n_feature))
    pairs = [(x, y) for x, y in combinations(nodes, 2)]
    sep_set = {}
    d = -1
    while _loop(skeleton, d):  # until for each adj(C,i)\{j} < l
        d += 1
        if variant == 'stable':
            C = deepcopy(skeleton)
        else:
            C = skeleton
        if variant != 'parallel':
            for i, j in pairs:
                if skeleton[i, j] == 0:
                    continue
                adj_i = set(np.argwhere(C[i] == 1).reshape(-1, ))
                z = adj_i - {j}  # adj(C, i)\{j}
                if len(z) >= d:
                    # |adj(C, i)\{j}| >= l
                    for sub_z in combinations(z, d):
                        sub_z = list(sub_z)
                        _, _, p_value = ci_test(data, i, j, sub_z)
                        if p_value >= alpha:
                            skeleton[i, j] = skeleton[j, i] = 0
                            sep_set[(i, j)] = sub_z
                            break
        else:
            J = [(x, y) for x, y in combinations(nodes, 2)
                 if skeleton[x, y] == 1]
            if not s or not batch:
                batch = len(J)
            if not p_cores or p_cores == 0:
                raise ValueError(f'If variant is parallel, type of p_cores '
                                 f'must be int, but got {type(p_cores)}.')
            for i in range(int(np.ceil(len(J) / batch))):
                each_batch = J[batch * i: batch * (i + 1)]
                parallel_result = joblib.Parallel(n_jobs=p_cores,
                                                  max_nbytes=None)(
                    joblib.delayed(parallel_cell)(x, y) for x, y in
                    each_batch
                )
                # Synchronisation Step
                for (x, y), K_x_y, sub_z in parallel_result:
                    if K_x_y == 0:
                        skeleton[x, y] = skeleton[y, x] = 0
                        sep_set[(x, y)] = sub_z

    return skeleton, sep_set


class PC(BaseLearner):
    """PC algorithm

    A classic causal discovery algorithm based on conditional independence tests.

    References
    ----------
    [1] original-PC
        https://www.jmlr.org/papers/volume8/kalisch07a/kalisch07a.pdf
    [2] stable-PC
        https://arxiv.org/pdf/1211.3295.pdf
    [3] parallel-PC
        https://arxiv.org/pdf/1502.02454.pdf

    Parameters
    ----------
    variant : str
        A variant of PC-algorithm, one of [`original`, `stable`, `parallel`].
    alpha: float, default 0.05
        Significance level.
    ci_test : str, callable
        ci_test method, if str, must be one of [`gauss`, `g2`, `chi2`]
        See more: `castle.common.independence_tests.CITest`

    Attributes
    ----------
    causal_matrix : array
        Learned causal structure matrix.

    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import load_dataset

    >>> X, true_dag, _ = load_dataset(name='IID_Test')
    >>> pc = PC(variant='stable')
    >>> pc.learn(X)
    >>> GraphDAG(pc.causal_matrix, true_dag, save_name='result_pc')
    >>> met = MetricsDAG(pc.causal_matrix, true_dag)
    >>> print(met.metrics)

    >>> pc = PC(variant='parallel')
    >>> pc.learn(X, p_cores=2)
    >>> GraphDAG(pc.causal_matrix, true_dag, save_name='result_pc')
    >>> met = MetricsDAG(pc.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self, variant='original', alpha=0.05, ci_test='gauss'):
        super(PC, self).__init__()
        self.variant = variant
        self.alpha = alpha
        self.ci_test = ci_test
        self.causal_matrix = None

    def learn(self, data, columns=None, **kwargs):
        """Set up and run the PC algorithm.

        Parameters
        ----------
        data: array or Tensor
            Training data
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        kwargs: [optional]
            p_cores : int
                number of CPU cores to be used
            s : boolean
                memory-efficient indicator
            batch : int
                number of edges per batch
            if s is None or False, or without batch, batch=|J|.
            |J| denote number of all pairs of adjacency vertices (X, Y) in G.
        """

        data = Tensor(data, columns=columns)

        skeleton, sep_set = find_skeleton(data,
                                          alpha=self.alpha,
                                          ci_test=self.ci_test,
                                          variant=self.variant,
                                          **kwargs)

        self._causal_matrix = Tensor(orient(skeleton, sep_set).astype(int),
                                     index=data.columns,
                                     columns=data.columns)

