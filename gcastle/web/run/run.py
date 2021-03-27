# encoding=utf-8
import os
import threading
import joblib
import numpy as np
from loguru import logger

from web.models.base_class import DataSetApi
from web.run.run_alg import run
from web.run.run_data_generation import simulate_data
from web.common.config import INLINE_ALGORITHMS, INLINE_DATASETS

from castle.common.base import Tensor


CHECK_INLINE_ALGORITHMS = [alg.lower() for alg, _ in INLINE_ALGORITHMS.items()]


def run_task(task_id, dataset, algorithm, parameters=None):
    """

    Parameters
    ----------
    task_id
    dataset
    algorithm
    parameters

    Returns
    -------

    """
    if DataSetApi.check_dataset(dataset):
        path = dataset
    elif dataset in INLINE_DATASETS:
        path = INLINE_DATASETS[dataset]
    elif dataset in DataSetApi.get_inline_dataset_names():
        try:
            dataset_task_id = dataset.split("_")[0]
            file_name = "sample_" + dataset_task_id + ".csv"
            file_path = os.path.join(DataSetApi.get_dataset(dataset_task_id),
                                     file_name)
            sample = np.loadtxt(file_path, delimiter=",")
            file_name = "true_dag_" + dataset_task_id + ".csv"
            file_path = os.path.join(DataSetApi.get_dataset(dataset_task_id),
                                     file_name)
            true_dag = np.loadtxt(file_path, delimiter=",")

            file_name = "topo_" + dataset_task_id + ".pkl"
            file_path = os.path.join(DataSetApi.get_dataset(dataset_task_id),
                                     file_name)
            topo = None
            if os.path.exists(file_path):
                topo = joblib.load(file_path)

            path = (true_dag, Tensor(sample), topo)
        except OSError as error:
            logger.warning('alg run fail %s' % str(error))
            return False
    else:
        return False
    if algorithm.lower() not in CHECK_INLINE_ALGORITHMS:
        return False
    else:
        # subprocess.Popen(cmd %task_id, shell=True)
        thread = threading.Thread(target=run,
                                  kwargs={"data": path,
                                          "alg": algorithm.lower(),
                                          "task_id": task_id,
                                          "header": 0,
                                          "parameters": parameters})
        thread.start()
        return True


def run_data(task_id, dataset, algorithm, parameters=None):
    """

    Parameters
    ----------
    task_id
    dataset
    algorithm
    parameters

    Returns
    -------

    """
    thread = threading.Thread(target=simulate_data,
                              kwargs={"data": dataset,
                                      "alg": algorithm,
                                      "task_id": task_id,
                                      "parameters": parameters})
    thread.start()
    return True
