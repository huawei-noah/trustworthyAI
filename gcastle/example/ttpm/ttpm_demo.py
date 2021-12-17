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
how to use TTPM algorithm in `castle` package for causal inference.

Warnings: This script is used only for demonstration and cannot be directly
        imported.
"""


from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, Topology, THPSimulation
from castle.algorithms import TTPM


# Data Simulation for TTPM
true_causal_matrix = DAG.erdos_renyi(n_nodes=10, n_edges=10)
topology_matrix = Topology.erdos_renyi(n_nodes=20, n_edges=20)
simulator = THPSimulation(true_causal_matrix, topology_matrix,
                          mu_range=(0.00005, 0.0001),
                          alpha_range=(0.005, 0.007))
X = simulator.simulate(T=3600*24, max_hop=2)

# TTPM modeling
ttpm = TTPM(topology_matrix, max_hop=2)
ttpm.learn(X)
print(ttpm.causal_matrix)

# plot est_dag and true_dag
GraphDAG(ttpm.causal_matrix, true_causal_matrix)
# calculate accuracy
ret_metrix = MetricsDAG(ttpm.causal_matrix, true_causal_matrix)
print(ret_metrix.metrics)
