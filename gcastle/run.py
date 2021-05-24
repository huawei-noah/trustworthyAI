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


import yaml
import argparse
import numpy as np
import pandas as pd

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import load_dataset


parser = argparse.ArgumentParser(description='Generic runner')
parser.add_argument('--model', '-m',
                    dest='model_name',
                    choices=['pc', 'gran_dag', 'direct_lingam', 'ica_lingam',
                             'notears', 'notears_mlp', 'notears_sob',
                             'notears_low_rank', 'notears_golem',
                             'gae', 'mcsl', 'rl', 'corl1', 'corl2', 'ttpm'],
                    help='name of algorithm',
                    default='rl')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='example/rl/rl.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        params_config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

x_file = params_config['dataset_params']['x_file']
dag_file = params_config['dataset_params']['dag_file']

if x_file != 'None':
    if args.model_name == 'ttpm':
        if x_file[-4:] == '.csv':
            X = pd.read_csv(x_file, header=0)
        else:
            raise ValueError('Invalid file type {}.'.format(x_file))

        topology_file = params_config['dataset_params']['topology_file']
        if topology_file[-4:] == '.npz':
            topology_data = np.load(topology_file)
            topology_matrix = np.mat(topology_data)
        elif topology_file[-4:] == '.csv':
            topology_data = np.loadtxt(topology_file, delimiter=',')
            topology_matrix = np.mat(topology_data)
        else:
            raise ValueError('Invalid file type {}.'.format(topology_file))
    else:
        if x_file[-4:] == '.npz':
            X = np.load(x_file)
        elif x_file[-4:] == '.csv':
            X = np.loadtxt(x_file, delimiter=',')
        else:
            raise ValueError('Invalid file type {}.'.format(x_file))

    if dag_file == 'None':
        true_dag = None
    else:
        if dag_file[-4:] == '.npz':
            true_dag = np.load(dag_file)
        elif dag_file[-4:] == '.csv':
            true_dag = np.loadtxt(dag_file, delimiter=',')
        else:
            raise ValueError('Invalid file type {}.'.format(dag_file))
else:
    if args.model_name == 'ttpm':
        true_dag, topology_matrix, X = load_dataset(name='thp_test')
    else:
        true_dag, X = load_dataset(name='iid_test')

# Instantiation algorithm and learn dag
if args.model_name == 'gran_dag':
    from castle.algorithms import GraN_DAG

    g = GraN_DAG(**params_config['model_params'])
    g.learn(data=X)

elif args.model_name == 'pc':
    from castle.algorithms import PC

    g = PC(**params_config['model_params'])
    g.learn(X)

elif args.model_name == 'notears':
    from castle.algorithms import Notears

    g = Notears(**params_config['model_params'])
    g.learn(data=X)

elif args.model_name == 'rl':
    from castle.algorithms import RL

    g = RL(**params_config['model_params'])
    g.learn(data=X, dag=true_dag)

elif args.model_name == 'ttpm':
    from castle.algorithms import TTPM

    g = TTPM(topology_matrix, **params_config['model_params'])
    g.learn(X)

else:
    raise ValueError('Invalid algorithm name: {}.'.format(args.model_name))

# plot and evaluate predict_dag and true_dag
if true_dag is not None:
    if args.model_name == 'ttpm':
        GraphDAG(g.causal_matrix.values, true_dag)
        m = MetricsDAG(g.causal_matrix.values, true_dag)
        print(m.metrics)
    else:
        GraphDAG(g.causal_matrix, true_dag)
        m = MetricsDAG(g.causal_matrix, true_dag)
        print(m.metrics)

else:
    if args.model_name == 'ttpm':
        GraphDAG(g.causal_matrix.values)
    else:
        GraphDAG(g.causal_matrix)
