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

from castle.datasets import DAG, IIDSimulation
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.algorithms.ges.ges import GES


for d in [6, 8, 10, 15, 20]:
    edges = d * 2
    weighted_random_dag = DAG.erdos_renyi(n_nodes=d, n_edges=edges,
                                          weight_range=(0.5, 2.0), seed=1)
    dataset = IIDSimulation(W=weighted_random_dag, n=1000,
                            method='nonlinear', sem_type='gp-add')
    true_dag, X = dataset.B, dataset.X

    algo = GES(criterion='bic', method='scatter')
    algo.learn(X)

    # plot predict_dag and true_dag
    GraphDAG(algo.causal_matrix, true_dag)
    m1 = MetricsDAG(algo.causal_matrix, true_dag)
    print(m1.metrics)
    break



