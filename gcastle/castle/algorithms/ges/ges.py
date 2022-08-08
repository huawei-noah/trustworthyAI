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

import numpy as np

from ...common import BaseLearner, Tensor
from .operators import search
from .score.local_scores import (BICScore, BDeuScore, DecomposableScore)


class GES(BaseLearner):
    """
    Greedy equivalence search for causal discovering

    References
    ----------
    [1]: https://www.sciencedirect.com/science/article/pii/S0888613X12001636
    [2]: https://www.jmlr.org/papers/volume3/chickering02b/chickering02b.pdf

    Parameters
    ----------
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

        Notes:
            1. 'bdeu' just for discrete variable.
            2. if you want to customize criterion, you must create a class
            and inherit the base class `DecomposableScore` in module
            `ges.score.local_scores`
    method: str
        effective when `criterion='bic'`, one of ['r2', 'scatter'].
    k: float, default: 0.001
        structure prior, effective when `criterion='bdeu'`.
    N: int, default: 10
        prior equivalent sample size, effective when `criterion='bdeu'`
    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import load_dataset

    >>> X, true_dag, _ = load_dataset(name='IID_Test')
    >>> algo = GES()
    >>> algo.learn(X)
    >>> GraphDAG(algo.causal_matrix, true_dag, save_name='result_pc')
    >>> met = MetricsDAG(algo.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    def __init__(self, criterion='bic', method='scatter', k=0.001, N=10):
        super(GES, self).__init__()
        if isinstance(criterion, str):
            if criterion not in ['bic', 'bdeu']:
                raise ValueError(f"if criterion is str, it must be one of "
                                 f"['bic', 'bdeu'], but got {criterion}.")
        else:
            if not isinstance(criterion, DecomposableScore):
                raise TypeError(f"The criterion is not instance of "
                                f"DecomposableScore.")
        self.criterion = criterion
        self.method = method
        self.k = k
        self.N = N

    def learn(self, data, columns=None, **kwargs):

        d = data.shape[1]
        e = np.zeros((d, d), dtype=int)

        if self.criterion == 'bic':
            self.criterion = BICScore(data=data,
                                      method=self.method)
        elif self.criterion == 'bdeu':
            self.criterion = BDeuScore(data=data, k=self.k, N=self.N)

        c = search.fes(C=e, criterion=self.criterion)
        c = search.bes(C=c, criterion=self.criterion)

        self._causal_matrix = Tensor(c, index=columns, columns=columns)
