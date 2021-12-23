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
import threading
import numpy as np
import pandas as pd
import logging as logger

from example.example import INLINE_ALGORITHMS, read_file
from web.common.config import INLINE_DATASETS

from web.models.base_class import DataSetApi
from web.run.run_alg import causal_discovery
from web.run.run_data_generation import simulate_data

CHECK_INLINE_ALGORITHMS = [alg for alg, _ in INLINE_ALGORITHMS.items()]


def run_task(task_id, dataset, algorithm, parameters=None):
    """
    Executing the Causal Discovery Algorithm Task.

    Parameters
    ----------
    task_id: int
        task key in the database.
    dataset: str
        data path.
    algorithm: str
        algorithm name.
    parameters: dict
        algorithm parameters.
    Returns
    -------
    : bool
        True: The causal discovery algorithm task is executed successfully.
        False: The cause-and-effect discovery algorithm task fails to be executed.
    """
    X = None
    true_dag = None
    topology_matrix = None

    if dataset in INLINE_DATASETS:
        try:
            file_path_list = dataset.split(os.path.sep)
            file_name = '.'.join(file_path_list[-1].split(".")[:-1])

            X = read_file(dataset, header=0)

            true_file_name = os.path.join(os.path.sep.join(file_path_list[:-2]), "true", file_name + ".npz")
            if os.path.exists(true_file_name):
                true_dag = read_file(true_file_name)

            topo_file_name = os.path.join(os.path.sep.join(file_path_list[:-2]), "topo_" + file_name + ".npz")
            if os.path.exists(topo_file_name):
                topology_matrix = read_file(topo_file_name)
        except OSError as error:
            logger.warning('alg run fail %s' % str(error))
            return False
    elif DataSetApi.check_dataset(dataset):
        if '.xls' in dataset:
            X = pd.read_excel(dataset, dtype=np.float64)
        elif '.csv' in dataset:
            X = pd.read_csv(dataset)
    else:
        return False

    if algorithm not in CHECK_INLINE_ALGORITHMS:
        return False
    else:
        thread = threading.Thread(target=causal_discovery,
                                  kwargs={"data": X,
                                          "true_dag": true_dag,
                                          "alg": algorithm,
                                          "algorithm_params": parameters,
                                          "task_id": task_id,
                                          'topology_matrix': topology_matrix})

        thread.start()
    return True


def run_data(task_id, dataset, algorithm, parameters=None):
    """
    Executing a Data Generation Task.

    Parameters
    ----------
    task_id: int
        task key in the database.
    dataset: str
        Directory for storing data files.
    algorithm:
        Generation operator.
    parameters
        Generate operator parameters.
    Returns
    -------
    : bool
    """
    thread = threading.Thread(target=simulate_data,
                              kwargs={"data": dataset,
                                      "alg": algorithm,
                                      "task_id": task_id,
                                      "parameters": parameters})
    thread.start()
    return True
