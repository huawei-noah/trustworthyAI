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
how to use CORL algorithm in `castle` package for causal inference.

If you want to plot causal graph, please make sure you have already install
`networkx` package, then like the following import method.

Warnings: This script is used only for demonstration and cannot be directly
          imported.
"""

import os
os.environ['CASTLE_BACKEND'] ='pytorch'

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import CORL

type = 'ER'  # or `SF`
h = 2  # ER2 when h=5 --> ER5
n_nodes = 10
n_edges = h * n_nodes
method = 'linear'
sem_type = 'gauss'

if type == 'ER':
    weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                          weight_range=(0.5, 2.0), seed=300)
elif type == 'SF':
    weighted_random_dag = DAG.scale_free(n_nodes=n_nodes, n_edges=n_edges,
                                         weight_range=(0.5, 2.0), seed=300)
else:
    raise ValueError('Just supported `ER` or `SF`.')

dataset = IIDSimulation(W=weighted_random_dag, n=2000,
                        method=method, sem_type=sem_type)
true_dag, X = dataset.B, dataset.X

# rl learn
rl = CORL(encoder_name='transformer',
          decoder_name='lstm',
          reward_mode='episodic',
          reward_regression_type='LR',
          batch_size=64,
          input_dim=64,
          embed_dim=64,
          iteration=2000,
          device_type='cpu')
rl.learn(X)

# plot est_dag and true_dag
GraphDAG(rl.causal_matrix, true_dag)

# calculate accuracy
met = MetricsDAG(rl.causal_matrix, true_dag)
print(met.metrics)
