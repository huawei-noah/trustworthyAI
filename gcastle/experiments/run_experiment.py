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
how to use RL algorithm in `castle` package for causal inference.

If you want to plot causal graph, please make sure you have already install
`networkx` package, then like the following import method.

Warnings: This script is used only for demonstration and cannot be directly
          imported.
"""

import os
import json
from argparse import ArgumentParser

import torch

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import RL


#######################################
# rl used simulate data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_nodes', default=10, type=int)
    parser.add_argument('--p', default=0.2, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_samples', default=5000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--nb_epoch', default=5000, type=int)
    parser.add_argument('--input_dimension', default=64, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--exponent_type', default='original')
    args = parser.parse_args()
    #######################################
    # simulate data for rl
    n_nodes = args.n_nodes
    n_edges = (args.p * n_nodes**2) // 2
    seed = args.seed

    weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges, weight_range=(0.5, 2.0), seed=seed)
    dataset = IIDSimulation(W=weighted_random_dag, n=args.num_samples, method='linear', sem_type='gauss')
    true_dag, X = dataset.B, dataset.X

    additional_kwargs = {}
    if args.exponent_type == 'trace_naive':
        additional_kwargs['m'] = 200

    # rl learn
    rl = RL(nb_epoch=args.nb_epoch,
            batch_size=args.batch_size,
            input_dimension=args.input_dimension,
            hidden_dim=args.hidden_dim,
            decoder_hidden_dim=args.hidden_dim,
            normalize=args.normalize,
            exponent_type=args.exponent_type,
            device_type='gpu' if torch.cuda.is_available() else 'cpu',
            **additional_kwargs)
    rl.learn(X)

    # plot est_dag and true_dag
    GraphDAG(rl.causal_matrix, true_dag)

    # calculate accuracy
    met = MetricsDAG(rl.causal_matrix, true_dag)
    os.makedirs('output', exist_ok=True)
    with open(f'output/{n_nodes}_{seed}_{args.exponent_type}.json', 'w') as f:
        json.dump(met.metrics, f)
    print(met.metrics)
