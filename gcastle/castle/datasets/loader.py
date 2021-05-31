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

from .simulation import DAG, IIDSimulation
from .simulation import Topology, THPSimulation


def load_dataset(name='iid_test'):
    """
    A function for loading some well-known datasets.

    Parameters
    ----------
    name: str, ('iid_test' or 'thp_test'), default='iid_test'
        Dataset name, independent and identically distributed (IID),
        Topological Hawkes Process (THP)

    Return
    ------
    out: tuple
        if name='iid_test':
            true_dag: np.ndarray
                adjacency matrix for the target causal graph.
            X: np.ndarray
                standard trainning dataset.
        if name='thp_test':
            true_dag: numpy.matrix
                adjacency matrix for the target causal graph.
            topology_matrix: numpy.matrix
                adjacency matrix for the topology.
            X: pandas.core.frame.DataFrame
                standard trainning dataset.
    """

    if name.lower() == 'iid_test':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=20, 
                                              weight_range=(0.5, 2.0),
                                              seed=1)
        dataset = IIDSimulation(W=weighted_random_dag, n=2000, 
                                method='linear', sem_type='gauss')
        true_dag, X = dataset.B, dataset.X

        return true_dag, X
        
    elif name.lower() == 'thp_test':
        true_dag = DAG.erdos_renyi(n_nodes=10, n_edges=10)
        topology_matrix = Topology.erdos_renyi(n_nodes=20, n_edges=20)
        simulator = THPSimulation(true_dag, topology_matrix, 
                                  mu_range=(0.00005, 0.0001),
                                  alpha_range=(0.005, 0.007))
        X = simulator.simulate(T=25000, max_hop=2)

        return true_dag, topology_matrix, X

    else:
        raise ValueError('The value of name must be iid_test or thp_test, '
                         'but got {}'.format(name))
