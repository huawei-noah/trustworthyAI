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

"""
This script is aim to make some experiments for current algorithms

dataset:
[Database with cause-effect pairs]
https://webdav.tuebingen.mpg.de/cause-effect/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import PNL


PATH = 'D:/Database/castle/cause_effect_pairs'


def load_x(number, encoding=None) -> np.array:
    """
    load x for data

    Parameters
    ----------
    number: int
        [1, 108]
    """

    no = '0' * (4-len(str(number))) + str(number)
    x_name = f"pair{no}.txt"

    # read x file
    with open(os.path.join(PATH, x_name), 'r', encoding=encoding) as xf:
        x = [xi.strip().split() for xi in xf.readlines()]
    x = np.array(x).astype('float')
    pt = PowerTransformer(standardize=False)
    x = pt.fit_transform(x)
    plt.hist(x[:, 0])
    plt.show()
    return x


def print_y(number, encoding=None):
    """
    print information of y

    Parameters
    ----------
    number: int
        [1, 108]
    """

    no = '0' * (4 - len(str(number))) + str(number)
    y_name = f"pair{no}_des.txt"

    # read y file
    with open(os.path.join(PATH, y_name), 'r', encoding=encoding) as yf:
        y = yf.readlines()[-2:]

    print(y)


def experiment_for_pair(number):
    x = load_x(number)
    model = PNL(batch_size=128)
    model.learn(x, columns=['x', 'y'])
    print(f'pair {number}==============================')
    print(model.causal_matrix)
    try:
        print_y(number)
    except UnicodeDecodeError:
        print_y(number, encoding='gbk')


def simulate_data(method='nonlinear', sem_type='mlp',
                  n_nodes=6, n_edges=15, n=1000):
    weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                          weight_range=(0.5, 2.0), seed=1)
    dataset = IIDSimulation(W=weighted_random_dag, n=n, method=method,
                            sem_type=sem_type)
    true_dag, X = dataset.B, dataset.X

    return X, true_dag


def castle_experiment(model, x, y=None, show_graph=False, **kwargs):

    model.learn(x, **kwargs)
    if y is not None:
        metrics = MetricsDAG(model.causal_matrix, y).metrics
    else:
        metrics = None
    if show_graph:
        GraphDAG(model.causal_matrix, y)

    return metrics



if __name__ == '__main__':

    # experiment for cause effect pair dataset
    # experiment_for_pair(1)


    # experiment for simulation data
    x, y = simulate_data(method='nonlinear', sem_type='gp',
                         n_nodes=6, n_edges=15, n=1000)
    # plt.hist(x[:, 5])
    # plt.show()
    model = PNL(batch_size=128)
    metrics = castle_experiment(model, x, y, show_graph=True)
    print(metrics)