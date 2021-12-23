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
import gettext
import operator
import os
import sys
import zipfile
import re
import json

import networkx as nx
import numpy as np
import pandas as pd
from inspect import isfunction, getfullargspec

from castle.datasets import DAG, IIDSimulation, THPSimulation, Topology
from example.example import INLINE_ALGORITHMS
from web.common.config import FILE_PATH, SEM_TYPE, INLINE_DATASETS, INLINE_TRUE

text = gettext.gettext


def zip_alg_file(task_id):
    """Packing the files related to the causal discovery task.

    Parameters
    ----------
    task_id: int
        task key in the database.
    Returns
    -------
    res: str
        zip file path.
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


def zip_data_file(task_id, task_name, data_path):
    """Files related to the packing data generation task.

    Parameters
    ----------
    data_path
    task_name
    task_id: int
        task key in the database.
    Returns
    -------
    zip_file_dir: str
        zip file path.
    """
    zip_file_dir = os.path.join(FILE_PATH, task_id + ".zip")
    file = zipfile.ZipFile(zip_file_dir, "w", zipfile.ZIP_DEFLATED)
    sample_path = os.path.join(data_path, "datasets", str(task_id) + "_" + task_name + ".csv")
    true_dag_path = os.path.join(data_path, "true", str(task_id) + "_" + task_name + ".npz")
    file.write(sample_path)
    file.write(true_dag_path)
    file.close()
    return zip_file_dir


def update_param(param, param_dict, alg="IID_LINEAR", prefix=""):
    """ Converting Function Parameters to Key-Value Pairs.

    Parameters
    ----------
    param: inspect.FullArgSpec
        Specifying formal arguments for a function.
    param_dict: dict
        Key-value pair of parameters and default values.
    alg: str
        Operator of the current task.
    prefix: str
        Prefix of different functions, which is used to distinguish functions with the same parameter name.
    Returns
    -------

    """
    default_len = len(param.defaults)
    if param.defaults:
        for index, value in enumerate(reversed(param.args)):
            if value not in ["self", "W", "method", "causal_matrix", "topology_matrix"]:
                if index < default_len:
                    p_value = list(reversed(param.defaults))[index]
                else:
                    p_value = None
                if value is "sem_type":
                    p_value = sem_type_set("sem_type", alg)[0]
                param_dict.update({prefix + value: p_value})


def simulate_parameters(alg="IID_LINEAR"):
    """Obtains parameters related to the data generation operator.

    Parameters
    ----------
    alg: str
        Operator of the current task.
    Returns
    -------
    param_dict: dict
        Key-value pairs of parameters related to the current operator.
    """
    param_dict = dict()

    param = getfullargspec(DAG.erdos_renyi)
    update_param(param, param_dict)

    if alg == "EVENT":
        param = getfullargspec(Topology.erdos_renyi)
        update_param(param, param_dict, prefix="Topology_")

        param = getfullargspec(THPSimulation.simulate)
        update_param(param, param_dict, prefix="THPSimulation_simulate_")

        param = getfullargspec(THPSimulation.__init__)
    else:
        param = getfullargspec(IIDSimulation.__init__)
    update_param(param, param_dict, alg, prefix="")
    return param_dict


def sem_type_set(cur_key, algorithm="IID_LINEAR"):
    """Obtains the option list of the sem_type parameter.

    Parameters
    ----------
    cur_key: str
        Parameter name string
    algorithm: str
        Task operator string
    Returns
    ----------
    res: list
        If the sem_type parameter is used, the parameter value list is returned. For other parameters, [] is returned.
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
    """Obtains parameters related to the causal discovery algorithm task.

    Parameters
    ----------
    alg: str
        Causal Discovery Operator String.

    Returns
    -------
    param_dict: dict
        Key-value pairs of parameters related to the current operator.
    """
    if alg in list(SEM_TYPE.keys()):
        return simulate_parameters(alg)

    param_dict = dict()

    param = getfullargspec(INLINE_ALGORITHMS[alg.upper()].__init__)
    if param is not None:
        param_len = len(param.args)
        if param.defaults:
            if 'input_dim' in param.args:
                param_dict.update({'input_dim': None})
            for index, value in enumerate(reversed(param.defaults)):
                if not isfunction(value) and (value is not None):
                    param_dict.update(
                        {param.args[param_len - index - 1]: value})
    param = getfullargspec(INLINE_ALGORITHMS[alg.upper()].learn)
    if param is not None:
        param_len = len(param.args)
        if param_len > 2:
            if 'rank' in param.args:
                param_dict.update({'rank': None})
    return param_dict


def conversion_type(alg, parameters):
    """Convert the parameter value type based on the default parameter type.

    Parameters
    ----------
    alg: str
        Causal Discovery Operator String.
    parameters: dict
        Key-value pairs of parameters related to the current operator.

    Returns
    -------
    parameters: dict
        Key-value pairs of parameters related to the current operator.
    """
    parameters = translation_parameters(parameters)
    if alg.upper() in INLINE_ALGORITHMS.keys():
        param = getfullargspec(INLINE_ALGORITHMS[alg.upper()].__init__)
        if param is not None:
            if param.defaults:
                for key, value in parameters.items():
                    if key != 'input_dim' and key != 'rank':
                        index = list(reversed(param.args)).index(key)
                        data_type = type(list(reversed(param.defaults))[index])(value)
                    else:
                        if value is None:
                            data_type = None
                        else:
                            data_type = int(value)
                    parameters[key] = data_type
    return parameters


def four_operations(formula):
    """four arithmetic operations

    Parameters
    ----------
    formula: string
    Four arithmetic strings
    Returns
    -------
    res: int
    Four Results of Operations
    """
    operators = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.or_}

    res = None
    for ops_str, ops in operators.items():
        if ops_str in formula:
            value_list = formula.split(ops_str)
            res = ops(int(value_list[0].strip()), int(value_list[1].strip()))
            break
    return res


def translation_parameters(parameters):
    """Converts the parameter value type as a string.

    Parameters
    ----------
    parameters: dict
        Key-value pairs of parameters related to the current operator.

    Returns
    -------
    param: dict
        Key-value pairs of parameters related to the current operator.
    """
    param = dict()
    if parameters:
        float_num = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        scientific_notation = re.compile(
            r"^[-+]?[1-9]?\.?[0-9]+[eE][-+]?[0-9]+$")
        int_num = re.compile(r'^[-+]?[0-9]+$')
        formula = re.compile(r'^(\s*)(.*)[0-9]+(\s*)(.*)[*+-\](\s*)(.*)[0-9]+(\s*)(.*)$')
        for key, value in parameters.items():
            if isinstance(value, str):
                value = value.strip()
                scientific_res = scientific_notation.match(value)
                float_res = float_num.match(value)
                int_res = int_num.match(value)
                formula_res = formula.match(value)
                if float_res or scientific_res:
                    value = float(value)
                elif int_res:
                    value = int(value)
                elif value.startswith("(") and value.endswith(")"):
                    value = tuple(json.loads(value))
                elif value.lower() in ["false", "true"]:
                    value = value.lower() == "true"
                elif formula_res:
                    value = four_operations(value)
                elif value == "":
                    value = None
            param.update({key: value})
    return param


def write_result(diagram, task_id):
    """data persistence.

    Parameters
    ----------
    diagram: str
        Formats of various objects converted to strings.
    task_id: int
        task key in the database.
    Returns
    -------

    """
    result_path = os.path.join(FILE_PATH, 'task', task_id)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(os.path.join(result_path, 'pre.txt'), 'w') as res_file:
        res_file.write(diagram)


def save_to_file(data, file):
    """save data to file

    Parameters
    ----------
    data: array or pd.DataFrame
        The data need to save.
    file: str
        where to save the data.
    """
    file_path = os.path.abspath(os.path.dirname(file) + os.path.sep + ".")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_list = file.split(".")
    if isinstance(data, pd.DataFrame):
        file_list[-1] = "csv"
        data.to_csv(".".join(file_list), index=False)
    else:
        file_list[-1] = "npz"
        np.savetxt(".".join(file_list), data, delimiter=',')


def save_gragh_edges(dag, file_name):
    """

    Parameters
    ----------
    dag
    file_name

    Returns
    -------

    """
    if isinstance(dag, pd.DataFrame):
        tem_dag = dag.values
    else:
        tem_dag = dag
    gragh = nx.from_numpy_array(tem_dag)
    diagrams = list(gragh.edges)

    zero_col = np.where(~tem_dag.any(axis=0))[0]
    zero_row = np.where(~tem_dag.any(axis=1))[0]
    zero_nodes = list(set(zero_col).intersection(set(zero_row)))
    if zero_nodes:
        for zero_node in zero_nodes:
            diagrams.append((zero_node, len(gragh)))

    if not hasattr(dag, "columns"):
        cols = [str(col) for col in range(len(gragh))]
    else:
        cols = list(dag.columns)
    cols.append('-')
    edges = [[str(cols[diagram[0]]), str(cols[diagram[1]])] for diagram in diagrams if
             isinstance(diagram, tuple) and len(diagram) > 1]

    with open(file_name, 'w') as res_file:
        res_file.write(json.dumps(edges))


def set_current_language(language):
    """

    Returns
    -------

    """
    localedir = os.path.dirname(os.path.abspath(sys.argv[0])) + '/locales'
    zh_CN = gettext.translation('base', localedir, languages=['zh_CN']);  # zh_CN en_US
    zh_CN.install()
    en_US = gettext.translation('base', localedir, languages=['en_US']);  # zh_CN en_US
    global text
    if language == "en_US":
        en_US.install()
        text = en_US.gettext
        return True
    elif language == "zh_CN":
        zh_CN.install()
        text = zh_CN.gettext
        return True
    return False


def _(id):
    """

    Parameters
    ----------
    id

    Returns
    -------

    """
    global text
    return text(id)


def update_inline_datasets():
    result_path = os.path.join(FILE_PATH, 'inline', 'datasets')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    datasets = os.listdir(result_path)
    for data in datasets:
        file_name = os.path.join(result_path, data)
        if file_name not in INLINE_DATASETS:
            INLINE_DATASETS.append(file_name)

    result_path = os.path.join(FILE_PATH, 'inline', 'true')
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    true_dags = os.listdir(result_path)
    for data in true_dags:
        file_name = os.path.join(result_path, data)
        if file_name not in INLINE_TRUE:
            INLINE_TRUE.append(file_name)
