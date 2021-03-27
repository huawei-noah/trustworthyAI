# encoding=utf-8

import os
import json
import pandas as pd
import numpy as np
from sqlite3 import DatabaseError

from web.common.utils import algorithm_parameters
from web.models.models import get_session
from web.models.task_db import Task, TaskApi
from web.common.config import INLINE_ALGORITHMS


class DataSetApi(object):
    def __init__(self, name_path=None):
        self.name_path = name_path
        self.session = get_session()

    def __del__(self):
        self.session.close()

    @classmethod
    def get_inline_dataset_names(cls):
        """

        Returns
        -------

        """
        dataset_names = TaskApi().get_simulation_name()
        return dataset_names

    def set_dataset_info(self, task_id):
        """

        :return: True, task_id
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


        Parameters
        ----------
        path

        Returns
        -------

        """
        if os.path.isfile(path):
            if '.xls' in path:
                data_df = pd.read_excel(path, dtype=np.float64)
            elif '.csv' in path:
                data_df = pd.read_csv(path)
            else:
                return False

            data_type = str(data_df.dtypes.unique()[0])
            if len(data_df.dtypes.unique()) == 1 and ('float' in data_type or 'int' in data_type):
                return True

        return False

    @classmethod
    def get_dataset(cls, task_id):
        """

        :param task_id:
        :return: dataset name(inline) or dataset path(customize)
        """
        task = TaskApi().get_task(task_id)
        if task:
            dataset = task.dataset
            return dataset
        return None


class AlgorithmApi(object):
    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.session = get_session()

    def __del__(self):
        self.session.close()

    @classmethod
    def get_algorithm_names(cls):
        """

        Returns
        -------

        """
        algorithm_names = list(INLINE_ALGORITHMS.keys())
        return algorithm_names

    def set_algorithm_info(self, task_id):
        """
        create or update algorithm info of a task
        :param task_id:
        :return:
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

    # 获取参数配置项，配置项均为空值
    @classmethod
    def get_algorithm_params(cls, name):
        """

        Parameters
        ----------
        name

        Returns
        -------

        """
        params = algorithm_parameters(name)
        return params

    @classmethod
    def get_algorithm(cls, task_id):
        """

        :param task_id:
        :return: {'name': ***, 'parameters': {}}
        """
        task = TaskApi().get_task(task_id)
        if task.algorithm and task.parameters:
            algorithm = task.algorithm
            parameters = json.loads(task.parameters)
            return {'algorithm': algorithm, 'parameters': parameters}
        return {}
