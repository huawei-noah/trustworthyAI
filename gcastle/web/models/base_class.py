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
import json
import pandas as pd
import numpy as np
from sqlite3 import DatabaseError

from example.example import INLINE_ALGORITHMS
from web.common.utils import algorithm_parameters
from web.models.models import get_session
from web.models.task_db import Task, TaskApi


class DataSetApi(object):
    """
    Access the fields related to data generation in the database.

    Parameters
    ----------
    name_path : str
        Path for storing generated data.

    Attributes
    ----------
    session : class sessionmaker
        Database Connection Session.
    """
    def __init__(self, name_path=None):
        self.name_path = name_path
        self.session = get_session()

    def __del__(self):
        self.session.close()

    @classmethod
    def get_inline_dataset_names(cls):
        """
        Combine the task ID and task name into a built-in dataset and obtain the list of built-in dataset names.

        Returns
        -------

        """
        dataset_names = TaskApi().get_simulation_name()
        return dataset_names

    def set_dataset_info(self, task_id):
        """
        Update the dataset field of the database.

        Parameters
        ----------
        task_id: int
            task key in the database.
        Returns
        -------
             True or False
        """
        try:
            self.session.query(Task).filter(Task.task_id == task_id).update({
                Task.dataset: self.name_path
            })
            self.session.commit()
            return True
        except DatabaseError:
            return False

    @classmethod
    def check_dataset(cls, path):
        """
        Check whether the data file exists.

        Parameters
        ----------
        path: str
            Data File Name.

        Returns
        -------
            True or False
        """
        if os.path.isfile(path):
            if '.xls' in path:
                data_df = pd.read_excel(path, dtype=np.float64)
            elif '.csv' in path:
                data_df = pd.read_csv(path)
            else:
                return None

            data_types = map(str, data_df.dtypes.unique())
            is_not_number = lambda dt: not ('float' in dt or 'int' in dt)
            if len(list(filter(is_not_number, data_types))) == 0:
                return data_df.shape[1]
        return None

    @classmethod
    def get_dataset(cls, task_id):
        """
        Obtaining Data File Names.

        Parameters
        ----------
        task_id: int
            task key in the database.
        Returns
        -------
        dataset: str
            name (inline) or dataset path (customize)
        """
        task = TaskApi().get_task(task_id)
        if task:
            dataset = task.dataset
            return dataset
        return None


class AlgorithmApi(object):
    """
    Access algorithm-related fields in the database.

    Parameters
    ----------
    name: str
        algorithm name.
    params: str
        Serialized string of the dictionary.

    Attributes
    ----------
    session : class sessionmaker
        Database Connection Session.
    """
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.session = get_session()

    def __del__(self):
        self.session.close()

    @classmethod
    def get_algorithm_names(cls):
        """
        Obtains the list of all algorithm names.

        Returns
        -------
        algorithm_names: list
            List of all algorithm names
        """
        algorithm_names = list(INLINE_ALGORITHMS.keys())
        return algorithm_names

    def set_algorithm_info(self, task_id):
        """
        Create or update algorithm info of a task.

        Parameters
        ----------
        task_id: int
            task key in the database.
        Returns
        -------
        tuple (HTML status code, dict)
            dict, task info
        """
        try:
            self.session.query(Task).filter(Task.task_id == task_id).update({
                Task.algorithm: self.name,
                Task.parameters: self.params
            })

            self.session.commit()
            task = self.session.query(Task).filter(Task.task_id == task_id).first()
            task_info = {'task_id': task.task_id,
                         'task_name': task.task_name,
                         'dataset': task.dataset,
                         'algorithm': task.algorithm,
                         'parameters': json.loads(task.parameters),
                         }
            return 200, task_info
        except DatabaseError:
            return 400, {}

    @classmethod
    def get_algorithm_params(cls, name):
        """
        Obtaining Algorithm Parameter Configuration Items

        Parameters
        ----------
        name: str
            algorithm name.
        Returns
        -------
        params: dict
            Key-value pair of the parameter name and default value.
        """
        params = algorithm_parameters(name)
        return params

    @classmethod
    def get_algorithm(cls, task_id):
        """
        Obtain the algorithm and parameters of the task.

        Parameters
        ----------
        task_id: int
            task key in the database.
        Returns
        -------
            {'name': ***, 'parameters': {}}
        """
        task = TaskApi().get_task(task_id)
        if task.algorithm and task.parameters:
            algorithm = task.algorithm
            parameters = json.loads(task.parameters)
            return {'algorithm': algorithm, 'parameters': parameters}
        return {}
