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


import importlib
import numpy as np
import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

from castle.common import Tensor
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation


def run_algorithm(algo: str,
                  X: [np.ndarray, Tensor],
                  true_dag: [np.ndarray, Tensor],
                  iteration: int = 1000,
                  device_type: str = 'cpu'):
    """run algorithm"""

    module = importlib.import_module(f'.algorithms', 'castle')
    model = getattr(module, algo)
    model = model(device_type=device_type, iteration=iteration)
    model.learn(X)
    met = MetricsDAG(model.causal_matrix, true_dag)


if __name__ == '__main__':

    algorithm = 'CORL'
    nodes = 10
    edges = 2 * nodes
    func = 'ER'
    method = 'linear'
    sem_type = 'gauss'

    if func == 'ER':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=nodes, n_edges=edges,
                                              weight_range=(0.5, 2.0), seed=300)
    elif func == 'SF':
        weighted_random_dag = DAG.scale_free(n_nodes=nodes, n_edges=edges,
                                             weight_range=(0.5, 2.0), seed=300)
    else:
        raise ValueError('Just supported `ER` or `SF`.')

    dataset = IIDSimulation(W=weighted_random_dag, n=2000,
                            method=method, sem_type=sem_type)
    true_dag, X = dataset.B, dataset.X
    met = run_algorithm('CORL', X, true_dag)
    pass