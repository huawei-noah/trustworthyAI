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


def insert(x, y, T, C):
    """
    Insert operator

    (1) inserting the directed edge X -> Y.
    (2) for each t of T, directing the previously undirected edge between t
        and Y as t -> Y.

    Parameters
    ----------
    x: int
        node X
    y: int
        node Y
    T: set
        subset of the neighbours of Y but not adjacent to X
    C: numpy.ndarray
        CPDAG

    Returns
    -------
    out: numpy.ndarray
        new C
    """

    C[x, y] = 1
    if len(T) != 0:
        T = list(T)
        C[T, y] = 1
        C[y, T] = 0

    return C


def insert_validity(x, y, T, C) -> bool:
    """
    check whether an insert operator is valid

    Notes
    -----
        condition1: NAyx U T is clique
        condition2: every semi-directed path from x to y contains a node in NAyx U T

    Parameters
    ----------
    x: int
        node X
    y: int
        node Y
    T: set
        subset of the neighbours of Y but not adjacent to X
    C: numpy.ndarray
        CPDAG

    Returns
    -------
    out: bool
        if True denotes the operator is valid, else False.
    """

    na_yx = graph.neighbors(y, C) & graph.adjacent(x, C)
    na_yx_t = na_yx | T

    # condition1
    condition1 = graph.is_clique(na_yx_t, C)

    # condition2
    semi_paths = graph.semi_directed_path(y, x, C)
    condition2 = True
    for path in semi_paths:
        if len(set(path) & na_yx_t) == 0:
            condition2 = False
            break
    if condition1 and condition2:
        return True
    else:
        return False
