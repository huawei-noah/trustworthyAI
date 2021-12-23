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
import datetime
import numpy as np
import pandas as pd
import logging as logger

from castle.datasets import IIDSimulation, THPSimulation, DAG, Topology

from web.models.task_db import TaskApi
from web.common.utils import translation_parameters, save_to_file, save_gragh_edges


def simulate_data(data, alg, task_id, parameters):
    """
    Simulation Data Generation Entry.

    Parameters
    ----------
    data: str
        Path for storing generated data files.
    alg: str
        Generating Operator Strings.
    task_id: int
        task key in the database.
    parameters: dict
        Data generation parameters.
    Returns
    -------
        True or False
    """
    parameters = translation_parameters(parameters)
    task_api = TaskApi()
    start_time = datetime.datetime.now()
    task_api.update_task_status(task_id, 0.1)
    task_api.update_consumed_time(task_id, start_time)
    task_api.update_update_time(task_id, start_time)

    if not os.path.exists(data):
        os.makedirs(data)
    task_name = task_api.get_task_name(task_id)
    sample_path = os.path.join(data, "datasets", str(task_id) + "_" + task_name + ".csv")
    true_dag_path = os.path.join(data, "true", str(task_id) + "_" + task_name + ".npz")
    node_relationship_path = os.path.join(data, "node_relationship_" + str(task_id) + "_" + task_name + ".csv")
    topo_path = os.path.join(data, "topo_" + str(task_id) + "_" + task_name + ".npz")
    task_api.update_task_status(task_id, 0.2)
    task_api.update_consumed_time(task_id, start_time)

    topo = None
    try:
        if alg == "EVENT":
            true_dag = DAG.erdos_renyi(n_nodes=parameters['n_nodes'],
                                       n_edges=parameters['n_edges'],
                                       weight_range=parameters['weight_range'],
                                       seed=parameters['seed'])
            topo = Topology.erdos_renyi(n_nodes=parameters['Topology_n_nodes'],
                                        n_edges=parameters['Topology_n_edges'],
                                        seed=parameters['Topology_seed'])
            simulator = THPSimulation(true_dag, topo,
                                      mu_range=parameters['mu_range'],
                                      alpha_range=parameters['alpha_range'])
            sample = simulator.simulate(T=parameters['THPSimulation_simulate_T'],
                                        max_hop=parameters['THPSimulation_simulate_max_hop'],
                                        beta=parameters['THPSimulation_simulate_beta'])

            task_api.update_task_status(task_id, 0.5)
            task_api.update_consumed_time(task_id, start_time)
        else:

            weighted_random_dag = DAG.erdos_renyi(n_nodes=parameters['n_nodes'],
                                                  n_edges=parameters['n_edges'],
                                                  weight_range=parameters['weight_range'],
                                                  seed=parameters['seed'])
            dataset = IIDSimulation(W=weighted_random_dag,
                                    n=parameters['n'],
                                    method=parameters['method'],
                                    sem_type=parameters['sem_type'],
                                    noise_scale=parameters['noise_scale'])

            true_dag, sample = dataset.B, dataset.X
            sample = pd.DataFrame(sample)

            task_api.update_task_status(task_id, 0.5)
            task_api.update_consumed_time(task_id, start_time)
    except Exception as error:
        task_api.update_task_status(task_id, str(error))
        task_api.update_consumed_time(task_id, start_time)
        logger.warning('Generating simulation data failed, exp=%s' % error)
        if os.path.exists(sample_path):
            os.remove(sample_path)
        if os.path.exists(true_dag_path):
            os.remove(true_dag_path)
        if os.path.exists(node_relationship_path):
            os.remove(node_relationship_path)
        if os.path.exists(topo_path):
            os.remove(topo_path)
        return False

    if os.path.exists(topo_path):
        os.remove(topo_path)

    task_api.update_task_status(task_id, 0.6)
    task_api.update_consumed_time(task_id, start_time)

    save_to_file(sample, sample_path)
    save_to_file(true_dag, true_dag_path)
    if isinstance(topo, np.ndarray):
        save_to_file(topo, topo_path)

    # calculate accuracy
    save_gragh_edges(true_dag, node_relationship_path)
    task_api.update_task_status(task_id, 1.0)
    task_api.update_consumed_time(task_id, start_time)
    return True
