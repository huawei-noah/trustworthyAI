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


def graph_prunned_by_coef(graph_batch, X, th=0.3):
    """
    for a given graph, pruning the edge according to edge weights;
    linear regression for each causal regresison for edge weights and then thresholding
    :param graph_batch: graph
    :param X: dataset
    :return:
    """
    d = len(graph_batch)
    reg = LinearRegression()
    W = []

    for i in range(d):
        col = np.abs(graph_batch[i]) > 0.1
        if np.sum(col) <= 0.1:
            W.append(np.zeros(d))
            continue

        X_train = X[:, col]

        y = X[:, i]
        reg.fit(X_train, y)
        reg_coeff = reg.coef_

        cj = 0
        new_reg_coeff = np.zeros(d, )
        for ci in range(d):
            if col[ci]:
                new_reg_coeff[ci] = reg_coeff[cj]
                cj += 1

        W.append(new_reg_coeff)

    return np.float32(np.abs(W) > th)


def graph_prunned_by_coef_2nd(graph_batch, X, th=0.3):
    """
    for a given graph, pruning the edge according to edge weights;
    quadratic regression for each causal regresison for edge weights and then thresholding
    :param graph_batch: graph
    :param X: dataset
    :return:
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
        X_train_expand_names =  poly.get_feature_names()[1:]
        
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
                        if np.abs(reg_coeff[iii]) > th:
                            new_reg_coeff[ci] = 1.0
                            break              
                cj += 1
        W.append(new_reg_coeff)

    return W


def convert_graph_int_to_adj_mat(graph_int):
    # Convert graph int to binary adjacency matrix
    # TODO: Make this more readable
    return np.array([list(map(int, ((len(graph_int) - len(np.base_repr(curr_int))) * '0' + np.base_repr(curr_int))))
                     for curr_int in graph_int], dtype=int)
