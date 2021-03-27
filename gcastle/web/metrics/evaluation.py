# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     evaluation
   Description :
   Author :       hWX889027
   date：          2020/7/24
-------------------------------------------------
"""

import os
import numpy as np
from loguru import logger

from web.models.base_class import DataSetApi, AlgorithmApi
from web.models.task_db import TaskApi

from castle.metrics.evaluation import MetricsDAG


class Evaluation:
    """
    evaluation
    """

    def __init__(self):
        self.evaluation_metrics = ["fdr", "tpr", "fpr", "shd", "nnz"]
        task_api = TaskApi()
        self.generat_operators = task_api.get_simulation_name()

    def get_label_checkbox(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task_api = TaskApi()
        label = task_api.get_label(task_id)
        builtin = label and label in self.generat_operators
        return builtin

    @staticmethod
    def get_task_evaluation_metrics(task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task_api = TaskApi()
        performance = task_api.get_performance(task_id)
        res = None
        if performance:
            res = list(performance.keys())
        return res

    def get_evaluation_metrics(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task_evaluation_metrics = self.get_task_evaluation_metrics(task_id)
        if not task_evaluation_metrics:
            task_evaluation_metrics = list()
        return {"evaluation_list": self.evaluation_metrics,
                "chosen_evaluation": task_evaluation_metrics}

    def get_task_builtin_label(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task_api = TaskApi()
        label = task_api.get_label(task_id)
        if label and label in self.generat_operators:
            builtin_label = label
        elif DataSetApi.get_dataset(task_id) in DataSetApi.get_inline_dataset_names():
            builtin_label = DataSetApi.get_dataset(task_id)
        else:
            builtin_label = None
        return builtin_label

    def get_builtin_label(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task_builtin_label = self.get_task_builtin_label(task_id)
        return {"operators": self.generat_operators,
                "selected_operators": task_builtin_label}

    def get_task_customize_label(self, task_id):
        """

        Parameters
        ----------
        task_id

        Returns
        -------

        """
        task_api = TaskApi()
        label = task_api.get_label(task_id)
        if label and label not in self.generat_operators:
            customize_label = label
        else:
            customize_label = None
        return {"label_data_path": customize_label}

    @staticmethod
    def check_label_dataset(label_path):
        """

        Parameters
        ----------
        label_path

        Returns
        -------

        """
        res = False
        if os.path.exists(label_path):
            if os.path.getsize(label_path):
                res = True
        return res

    def evaluation_execute(self, task_id, label_path, chosen_evaluation):
        """

        Parameters
        ----------
        task_id
        label_path
        chosen_evaluation

        Returns
        -------

        """
        task_api = TaskApi()
        est_dag = task_api.get_est_dag(task_id)

        try:
            alg = AlgorithmApi.get_algorithm(task_id)["algorithm"]

            if alg == "TTPM":
                true_dag = task_api.get_true_dag(task_id)
            elif label_path in self.generat_operators:
                dataset_task_id = label_path.split("_")[0]
                file_name = "true_dag_" + dataset_task_id + ".csv"
                file_path = os.path.join(DataSetApi.get_dataset(dataset_task_id),
                                         file_name)
                true_dag = np.loadtxt(file_path, delimiter=",")
                task_api.update_true_dag(task_id, true_dag)
            else:
                true_dag = np.loadtxt(label_path, delimiter=",")
                task_api.update_true_dag(task_id, true_dag)

            metrics = MetricsDAG(est_dag, true_dag)
        except Exception as error:
            logger.warning('evaluation execute failed, exp=%s' % error)
            return {"status": 400, "data": str(error)}

        evaluation_metrics = dict()
        for evaluation, _ in metrics.metrics.items():
            if evaluation in chosen_evaluation:
                evaluation_metrics.update(
                    {evaluation: metrics.metrics[evaluation]})

        task_api = TaskApi()
        task_api.update_performance(task_id, label_path, evaluation_metrics)
        return {"task_id": task_id,
                "evaluations": evaluation_metrics}
