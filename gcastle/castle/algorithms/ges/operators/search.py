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

from ..functional import graph
from ..functional.utils import subset_generator
from .inserter import insert, insert_validity
from .deleter import delete, delete_validity


def fes(C, criterion):
    """
    Forward Equivalence Search

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: np.array
        cpdag
    """

    while True:
        edge, t = forward_search(C, criterion)
        if edge is None:
            break
        x, y = edge
        C = insert(x, y, t, C)
        C = graph.pdag_to_cpdag(C)

    return C


def forward_search(C, criterion):
    """
    forward search

    starts with an empty (i.e., no-edge) CPDAG and greedily applies GES
    insert operators until no operator has a positive score.

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: tuple
        ((X, Y), T), the edge (X, Y) denotes X->Y is valid and T is a subset of
        the neighbors of Y that are not adjacent to X,
    """

    d = C.shape[0]
    edge = None
    subset = {}
    best = 0
    V = np.arange(d)
    for x in V:
        Vy = graph.connect(x, C, relation=None)
        for y in Vy:
            T0 = subset_generator(graph.neighbors(y, C) - graph.adjacent(x, C))
            for T in T0:
                if not insert_validity(x, y, T, C):
                    continue
                # det = f (Y, PaPC (Y) ∪ {X} ∪ T ∪ NAY,X ) − f (Y, PaPC (Y) ∪ T ∪ NAY,X ).
                na_yx = graph.neighbors(y, C) & graph.adjacent(x, C)
                pa_y = graph.parent(y, C)
                pa1 = pa_y | {x} | T | na_yx
                pa2 = pa_y | T | na_yx
                try:
                    det = (criterion.local_score(y, pa1)
                           - criterion.local_score(y, pa2))
                except AttributeError:
                    raise AttributeError(f"The criterion has no attribute named "
                                         f"`local_score`, you can create a class inherit"
                                         f"`DecomposableScore` and implement `local_score`"
                                         f" method.")

                if det > best:
                    best = det
                    edge = (x, y)
                    subset = T
    return edge, subset


def bes(C, criterion):
    """
    Backward Equivalence Search

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: np.array
        cpdag
    """

    while True:
        edge, h = backward_search(C, criterion)
        if edge is None:
            break
        x, y = edge
        C = delete(x, y, h, C)
        C = graph.pdag_to_cpdag(C)

    return C


def backward_search(C, criterion):
    """
    backward search

    starts with a CPDAG and greedily applies GES delete operators until no
    operator has a positive score.

    Parameters
    ----------
    C: np.array
        [d, d], cpdag
    criterion: str for DecomposableScore object
        scoring criterion, one of ['bic', 'bdeu'].

    Returns
    -------
    out: tuple
        ((X, Y), H), the edge (X, Y) denotes X->Y is valid and H is a subset of
        the neighbors of Y that are adjacent to X,
    """

    d = criterion.d
    edge = None
    subset = {}
    best = 0
    V = np.arange(d)
    for x in V:
        Vy = graph.adjacent(x, C)
        for y in Vy:
            H0 = subset_generator(graph.neighbors(y, C) - graph.adjacent(x, C))
            for H in H0:
                if not delete_validity(x, y, H, C):
                    continue
                # det = f (Y, PaPC (Y) ∪ {NAY,X \ H} \ X) − f (Y, PaPC (Y) ∪ {NAY,X \ H}).
                na_yx = graph.neighbors(y, C) & graph.adjacent(x, C)
                pa_y = graph.parent(y, C)
                pa1 = pa_y | ((na_yx - H) - {x})
                pa2 = pa_y | (na_yx - H)
                det = (criterion.local_score(y, pa1)
                       - criterion.local_score(y, pa2))
                if det > best:
                    best = det
                    edge = (x, y)
                    subset = H

    return edge, subset
