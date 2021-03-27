# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     run_make_data
   Description :
   Author :       hWX889027
   date：          2020/11/24
-------------------------------------------------
"""
import os
import json
import datetime
import joblib
import numpy as np
import pandas as pd
from loguru import logger

from castle.common import GraphDAG
from castle.datasets import IIDSimulation, EventSimulation

from web.models.task_db import TaskApi
from web.common.utils import translation_parameters
from web.common.config import GRAPH_TYPE


def simulate_data(data, alg, task_id, parameters):
    """

    Parameters
    ----------
    data
    alg
    task_id
    parameters

    Returns
    -------

    """
    parameters = translation_parameters(parameters)
    task_api = TaskApi()
    start_time = datetime.datetime.now()
    task_api.update_task_status(task_id, 0.1)
    task_api.update_consumed_time(task_id, start_time)
    task_api.update_update_time(task_id, start_time)

    if not os.path.exists(data):
        os.makedirs(data)
    sample_path = os.path.join(data, "sample_" + task_id + ".csv")
    true_dag_path = os.path.join(data, "true_dag_" + task_id + ".csv")
    node_relationship_path = os.path.join(data,
                                          "node_relationship_" + task_id + ".csv")
    topo_path = os.path.join(data, "topo_" + task_id + ".pkl")
    task_api.update_task_status(task_id, 0.2)
    task_api.update_consumed_time(task_id, start_time)

    topo = None
    try:
        if alg == "EVENT":
            dataset = EventSimulation(**parameters)
            sample, topo, true_dag = dataset.event_table, dataset.topo, dataset.edge_mat
            task_api.update_task_status(task_id, 0.5)
            task_api.update_consumed_time(task_id, start_time)
        else:
            sem_type = parameters["sem_type"]
            num = parameters["n"]
            del parameters["sem_type"]
            del parameters["n"]
            dataset = IIDSimulation(W=GRAPH_TYPE[alg.split("_")[-1]](**parameters),
                                    n=num,
                                    method=alg.split("_")[-2], sem_type=sem_type)
            true_dag, sample = dataset.W, dataset.X
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

    gragh = GraphDAG.nx_graph(pd.DataFrame(true_dag))
    task_api.update_task_status(task_id, 0.6)
    task_api.update_consumed_time(task_id, start_time)

    np.savetxt(sample_path, sample, delimiter=",")
    np.savetxt(true_dag_path, true_dag, delimiter=",")
    if topo:
        joblib.dump(topo, topo_path)

    diagrams = list(gragh.edges)
    edges = [[str(diagram[0]), str(diagram[1])] for diagram in diagrams if
             isinstance(diagram, tuple) and len(diagram) > 1]
    with open(node_relationship_path, 'w') as res_file:
        res_file.write(json.dumps(edges))
    task_api.update_task_status(task_id, 1.0)
    task_api.update_consumed_time(task_id, start_time)
    return True
