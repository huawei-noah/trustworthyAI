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
This demo script aim to demonstrate
how to use PC algorithm in `castle` package for causal inference.

If you want to plot causal graph, please make sure you have already install
`networkx` package, then like the following import method.

Warnings: This script is used only for demonstration and cannot be directly
        imported.
"""

import pandas as pd
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import PC


method = 'linear'
sem_type = 'gauss'
n_nodes = 8
n_edges = 15
n = 2000

# simulation for pc
weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges, weight_range=(0.5, 2.0), seed=1)
dataset = IIDSimulation(W=weighted_random_dag, n=n, method=method, sem_type=sem_type)
true_dag, X = dataset.B, dataset.X
# PC learn
pc = PC(variant='original')
X = pd.DataFrame(X, columns=list('abcdefgh'))
pc.learn(X)

# plot predict_dag and true_dag
GraphDAG(pc.causal_matrix, true_dag)

# calculate accuracy
met = MetricsDAG(pc.causal_matrix, true_dag)
print(met.metrics)
