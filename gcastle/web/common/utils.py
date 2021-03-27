# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       hWX889027
   date：          2020/11/11
-------------------------------------------------
"""

import os
import zipfile
import re
from inspect import getargspec, isfunction, getfullargspec

from castle.algorithms.gradient.gran_dag import Parameters as GDP
from castle.algorithms.gradient.graph_auto_encoder import Parameters as GP
from castle.datasets import IIDSimulation, EventSimulation

from web.common.config import FILE_PATH, GRAPH_TYPE, SEM_TYPE, INLINE_ALGORITHMS


def zip_alg_file(task_id):
    """

    Parameters
    ----------
    task_id

    Returns
    -------

    """
    start_dir = os.path.join(FILE_PATH, "task", task_id)
    res = None
    if os.path.exists(start_dir):
        zip_file_dir = os.path.join(FILE_PATH, "task", task_id + ".zip")
        file = zipfile.ZipFile(zip_file_dir, "w", zipfile.ZIP_DEFLATED)
        for dir_path, _, file_names in os.walk(start_dir):
            for file_name in file_names:
                file.write(os.path.join(dir_path, file_name))
        file.close()
        res = zip_file_dir
    return res


def zip_data_file(task_id):
    """

    Parameters
    ----------
    task_id

    Returns
    -------

    """
    zip_file_dir = os.path.join(FILE_PATH, task_id + ".zip")
    file = zipfile.ZipFile(zip_file_dir, "w", zipfile.ZIP_DEFLATED)
    file_name = "sample_" + task_id + ".csv"
    file.write(os.path.join(FILE_PATH, file_name))
    file_name = "true_dag_" + task_id + ".csv"
    file.write(os.path.join(FILE_PATH, file_name))
    file.close()
    return zip_file_dir


def simulate_parameters(alg="IID_linear", graph="ER"):
    """

    Parameters
    ----------
    algorithm

    Returns
    -------

    """
    param_dict = dict()
    if alg != "EVENT":
        param = getfullargspec(GRAPH_TYPE[graph])
        param_len = len(param.args)
        if param.defaults:
            for index, value in enumerate(reversed(param.defaults)):
                if param.args[param_len - index - 1] is not "w_range":
                    param_dict.update(
                        {param.args[param_len - index - 1]: value})

    if alg == "EVENT":
        param = getfullargspec(EventSimulation.__init__)
    else:
        param = getfullargspec(IIDSimulation.__init__)
    param_len = len(param.args)
    if param.defaults:
        for index, value in enumerate(reversed(param.defaults)):
            if param.args[param_len - index - 1] not in ["W", "method",
                                                         "noise_scale"]:
                if param.args[param_len - index - 1] is "sem_type":
                    value = sem_type_set("sem_type", alg)[0]
                if not isinstance(value, tuple):
                    param_dict.update(
                        {param.args[param_len - index - 1]: value})
    return param_dict


def sem_type_set(cur_key, algorithm="IID_linear"):
    """

    Args:
        cur_key:
        algorithm:

    Returns:

    """
    if algorithm in list(SEM_TYPE.keys()):
        if cur_key == "sem_type":
            res = SEM_TYPE[algorithm]
        else:
            res = []
    else:
        res = []
    return res


def algorithm_parameters(alg):
    """

    Parameters
    ----------
    alg

    Returns
    -------

    """
    if alg in list(SEM_TYPE.keys()):
        return simulate_parameters(alg)

    param_dict = dict()

    if alg == "GAE":
        param = getfullargspec(GP.__init__)
    elif alg == "GRAN":
        param = getfullargspec(GDP.__init__)
    else:
        alg_obj = INLINE_ALGORITHMS[alg]()
        if hasattr(alg_obj, "learn"):
            param = getargspec(alg_obj.learn)

    if param is not None:
        param_len = len(param.args)
        if param.defaults:
            for index, value in enumerate(reversed(param.defaults)):
                if not isfunction(value) and (value is not None):
                    param_dict.update(
                        {param.args[param_len - index - 1]: value})
    return param_dict


def translation_parameters(parameters):
    """

    Parameters
    ----------
    parameters

    Returns
    -------

    """
    param = dict()
    if parameters:
        float_num = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        scientific_notation = re.compile(
            r"^[-+]?[1-9]?\.?[0-9]+[eE][-+]?[0-9]+$")
        int_num = re.compile(r'^[-+]?[0-9]+$')
        for key, value in parameters.items():
            if isinstance(value, str):
                scientific_res = scientific_notation.match(value)
                float_res = float_num.match(value)
                int_res = int_num.match(value)
                if float_res or scientific_res:
                    value = float(value)
                elif int_res:
                    value = int(value)
                elif value.lower() in ["false", "true"]:
                    value = value.lower() == str(True)
            param.update({key: value})
    return param


def write_result(diagram, task_id):
    """

    Parameters
    ----------
    diagram
    task_id

    Returns
    -------

    """
    result_path = os.path.join(FILE_PATH, 'task', task_id)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(os.path.join(result_path, task_id + '.txt'), 'w') as res_file:
        res_file.write(diagram)
