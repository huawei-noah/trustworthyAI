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
import logging as logger
import numpy as np

from example.example import train
from web.common.config import FILE_PATH

from web.models.task_db import TaskApi
from web.common.utils import conversion_type, save_gragh_edges


def causal_discovery(data, true_dag, alg='PC', algorithm_params=None, task_id=None, topology_matrix=None):
    """
    Path for executing causal discovery, and update list status.

    Parameters
    ----------
    data: pd.DataFrame or array
        Raw data.
    true_dag: array
        Realistic map.
    alg: str
        Causal discovery algorithm.
    algorithm_params: dict
        Causal Discovery Algorithm Parameters.
    task_id: int
        task key in the database.
    topology_matrix: numpy.ndarray
        topology matrix only TTPM used.
    Returns
    -------

    """
    task_api = TaskApi()
    try:
        start_time = datetime.datetime.now()
        task_api.update_performance(task_id, "", dict())
        task_api.update_task_status(task_id, 0.1)
        task_api.update_consumed_time(task_id, start_time)
        task_api.update_update_time(task_id, start_time)
        # algorithm
        algorithm_params = conversion_type(alg, algorithm_params)
        p_res, pre_dag = train(alg, data, true_dag, algorithm_params, topology_matrix=topology_matrix, plot=False)

        task_api.update_task_status(task_id, 0.5)
        task_api.update_consumed_time(task_id, start_time)

        task_api.update_est_dag(task_id, p_res.causal_matrix)
        # pre_dag = p_res.causal_matrix

        task_path = os.path.join(FILE_PATH, 'task', task_id)
        if not os.path.exists(task_path):
            os.makedirs(task_path)

        if isinstance(true_dag, np.ndarray):
            task_api.update_true_dag(task_id, true_dag)
            file_name = os.path.join(task_path, "true.txt")
            save_gragh_edges(true_dag, file_name)

        task_api.update_task_status(task_id, 0.8)
        task_api.update_consumed_time(task_id, start_time)

        # deal result
        file_name = os.path.join(task_path, "pre.txt")
        save_gragh_edges(pre_dag, file_name)
        task_api.update_task_status(task_id, 1.0)
        task_api.update_consumed_time(task_id, start_time)
    except Exception as error:
        task_api.update_task_status(task_id, str(error))
        task_api.update_consumed_time(task_id, start_time)
        logger.warning('alg run fail %s' % str(error))
