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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def get_graph_from_order(sequence, dag_mask=None) -> np.ndarray:
    """
    Generate a fully-connected DAG based on a sequence.


    Parameters
    ----------
    sequence: iterable
        An ordering of nodes, the set of nodes that precede node vj
        denotes potencial parent nodes of vj.
    dag_mask : ndarray
        two-dimensional array with [0, 1], shape = [n_nodes, n_nodes].
        (i, j) indicated element `0` denotes there must be no edge
        between nodes `i` and `j` , the element `1` indicates that
        there may or may not be an edge.

    Returns
    -------
    out:
        graph matrix

    Examples
    --------
    >>> order = [2, 0, 1, 3]
    >>> graph = get_graph_from_order(sequence=order)
    >>> print(graph)
        [[0. 1. 0. 1.]
         [0. 0. 0. 1.]
         [1. 1. 0. 1.]
         [0. 0. 0. 0.]]
    """

    num_node = len(sequence)
    init_graph = np.zeros((num_node, num_node))
    for i in range(num_node - 1):
        pa_node = sequence[i]
        sub_node = sequence[i + 1:]
        init_graph[pa_node, sub_node] = 1
    if dag_mask is None:
        gtrue_mask = np.ones([num_node, num_node]) - np.eye(num_node)
    else:
        gtrue_mask = dag_mask
    dag_mask = np.int32(np.abs(gtrue_mask) > 1e-3)
    init_graph = init_graph * dag_mask

    return init_graph


def cover_rate(graph, graph_true) -> np.ndarray:

    error = graph - graph_true

    return np.sum(np.float32(error > -0.1))


def pruning_by_coef(graph_batch, X, thresh=0.3) -> np.ndarray:
    """
    for a given graph, pruning the edge according to edge weights;
    linear regression for each causal regression for edge weights and
    then thresholding
    """

    n, d = X.shape
    reg = LinearRegression()
    W = []

    loss = 0

    for i in range(d):
        col = np.abs(graph_batch[i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]

        y = X[:, i]
        reg.fit(X_train, y)
        loss += 0.5 / n * np.sum(np.square(reg.predict(X_train) - y))
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1

        W.append(new_reg_coeff)

    return np.float32(np.abs(W) > thresh)


def pruning_by_coef_2nd(graph_batch, X, thresh=0.3) -> np.ndarray:
    """
    for a given graph, pruning the edge according to edge weights;
    quadratic regression for each causal regression for edge weights and then
    thresholding
    """

    d = len(graph_batch)
    reg = LinearRegression()
    poly = PolynomialFeatures()
    W = []

    for i in range(d):
        col = graph_batch[i] > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]
        X_train_expand = poly.fit_transform(X_train)[:, 1:]
        X_train_expand_names = poly.get_feature_names()[1:]

        y = X[:, i]
        reg.fit(X_train_expand, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )

        for ci in range(d):
            if col[ci]:
                xxi = 'x{}'.format(cj)
                for iii, xxx in enumerate(X_train_expand_names):
                    if xxi in xxx:
                        if np.abs(reg_coeff[iii]) > thresh:
                            new_reg_coeff[ci] = 1.0
                            break
                cj += 1
        W.append(new_reg_coeff)

    return np.array(W)

