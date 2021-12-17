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

import os
import numpy as np
import pandas as pd

from castle.algorithms import PC
from castle.algorithms import ANMNonlinear
from castle.algorithms import TTPM
from castle.algorithms import DirectLiNGAM, ICALiNGAM
from castle.algorithms import GraNDAG, CORL, RL
from castle.algorithms import NotearsLowRank, NotearsNonlinear, Notears
from castle.algorithms import MCSL, GOLEM
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation, Topology, THPSimulation


INLINE_ALGORITHMS = {
    "DIRECTLINGAM": DirectLiNGAM,
    "ICALINGAM": ICALiNGAM,
    "PC": PC,
    "NOTEARS": Notears,
    "NOTEARSLOWRANK": NotearsLowRank,
    "NOTEARSNONLINEAR": NotearsNonlinear,
    "ANMNONLINEAR": ANMNonlinear,
    "GRANDAG": GraNDAG,
    "RL": RL,
    "CORL": CORL,
    "TTPM": TTPM,
    "MCSL": MCSL,
    "GOLEM": GOLEM
}


def save_to_file(data, file):
    """save data to file

    Parameters
    ----------
    data: array or pd.DataFrame
        The data need to save.
    file: str
        where to save the data.
    """

    if file.split('.')[-1] in ['csv', 'npz']:
        if isinstance(data, pd.DataFrame):
            data.to_csv(file, index=False)
        else:
            np.savetxt(file, data, delimiter=',')
    else:
        raise ValueError(f'Invalid file type : {file}, muse be csv.')


def run_simulate(config):
    """this function used to run simulate data task

    Parameters
    ----------
    config: dict
        configuration info.

    Returns
    -------
    out: tuple
        (X, true_dag) or (X, true_dag, topology_matrix)
    """

    algo_params = config['algorithm_params']
    if config['task_params']['algorithm'] == 'EVENT':
        true_dag = DAG.erdos_renyi(n_nodes=algo_params['n_nodes'],
                                   n_edges=algo_params['n_edges'],
                                   weight_range=algo_params['weight_range'],
                                   seed=algo_params['seed'])
        topology_matrix = Topology.erdos_renyi(n_nodes=algo_params['Topology_n_nodes'],
                                               n_edges=algo_params['Topology_n_edges'],
                                               seed=algo_params['Topology_seed'])
        simulator = THPSimulation(true_dag, topology_matrix,
                                  mu_range=algo_params['mu_range'],
                                  alpha_range=algo_params['alpha_range'])
        X = simulator.simulate(T=algo_params['THPSimulation_simulate_T'],
                               max_hop=algo_params['THPSimulation_simulate_max_hop'],
                               beta=algo_params['THPSimulation_simulate_beta'])

        return X, true_dag, topology_matrix
    else:
        weighted_random_dag = DAG.erdos_renyi(n_nodes=algo_params['n_nodes'],
                                              n_edges=algo_params['n_edges'],
                                              weight_range=algo_params['weight_range'],
                                              seed=algo_params['seed'])
        dataset = IIDSimulation(W=weighted_random_dag,
                                n=algo_params['n'],
                                method=algo_params['method'],
                                sem_type=algo_params['sem_type'],
                                noise_scale=algo_params['noise_scale'])

        return pd.DataFrame(dataset.X), dataset.B


def read_file(file, header=None, index_col=None):
    """read data from file

    Parameters
    ----------
    file: str
        file path of data.
    header: None or int
        Be used in pd.read_csv.
        If file type is `.csv`, you must provide right header to make sure
        contain all examples dataset.
    index_col: None or int
        Be used in pd.read_csv.
        If file type is `.csv`, you must provide right index_col to make sure
        contain all examples dataset.

    Returns
    -------
    out: array
        data set.
    """

    if file and file[-4:] == '.csv' and os.path.exists(file):
        x = pd.read_csv(file, header=header, index_col=index_col)
    elif file and file[-4:] == '.npz' and os.path.exists(file):
        x = np.loadtxt(file, delimiter=",")
    else:
        raise ValueError('Invalid file type {}.'.format(file))

    return x


def train(model_name, X, true_dag, model_params, topology_matrix=None, plot=True):
    """run algorithm of castle

    Parameters
    ----------
    model_name: str
        algorithm name
    X: pd.DataFrame
        train data
    true_dag: array
        true directed acyclic graph
    model_params: dict
        Parameters from configuration file
    topology_matrix: array, default None
        topology graph matrix
    plot: boolean, default None
        whether show graph.

    Returns
    -------
    model: castle.algorithm
        model of castle.algorithm
    pre_dag: array
        discovered causal matrix
    """

    # Instantiation algorithm and learn dag
    if model_name == 'TTPM':
        model = INLINE_ALGORITHMS[model_name.upper()](topology_matrix, **model_params)
        model.learn(X)
    elif model_name == 'NOTEARSLOWRANK':
        rank = model_params.get('rank')
        del model_params['rank']
        model = NotearsLowRank(**model_params)
        model.learn(X, rank=rank)
    else:
        try:
            model = INLINE_ALGORITHMS[model_name.upper()](**model_params)
            model.learn(data=X)
        except ValueError:
            raise ValueError('Invalid algorithm name: {}.'.format(model_name))

    pre_dag = model.causal_matrix
    if plot:
        if true_dag is not None:
            GraphDAG(pre_dag, true_dag, show=plot)
            m = MetricsDAG(pre_dag, true_dag)
            print(m.metrics)
        else:
            GraphDAG(pre_dag, show=plot)

    return model, pre_dag
