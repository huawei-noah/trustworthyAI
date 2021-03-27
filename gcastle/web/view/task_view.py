# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     task
   Description :
   Author :       hWX889027
   date：          2020/8/7
-------------------------------------------------
"""

import json
import os
from flask import Blueprint, make_response, send_file, request, jsonify

from web.metrics.evaluation import Evaluation
from web.models.base_class import DataSetApi, AlgorithmApi
from web.models.task_db import TaskApi
from web.run import run
from web.common.config import SEM_TYPE, GRAPH_TYPE, FILE_PATH
from web.common.utils import simulate_parameters, algorithm_parameters, \
    zip_alg_file, zip_data_file, sem_type_set, write_result

task = Blueprint("task", __name__)


@task.route("/get_task_list", methods=["GET"])
def get_task_list():
    """

    Returns:

    """
    task_api = TaskApi()
    tasks = task_api.list_tasks()
    return jsonify(tasks)


@task.route("/get_label_checkbox", methods=["POST"])
def get_label_checkbox():
    """

    Returns:

    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    is_inline = evaluation.get_label_checkbox(int(task_id))
    return jsonify({"is_inline": str(is_inline)})


@task.route("/get_evaluation_metrics", methods=["POST"])
def get_evaluation_metrics():
    """

    Returns:

    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    return jsonify(evaluation.get_evaluation_metrics(int(task_id)))


@task.route("/get_builtin_label", methods=["POST"])
def get_builtin_label():
    """

    Returns:

    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    return jsonify(evaluation.get_builtin_label(int(task_id)))


@task.route("/get_customize_label", methods=["POST"])
def get_customize_label():
    """

    Returns:

    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    return jsonify(evaluation.get_task_customize_label(int(task_id)))


@task.route("/check_label_dataset", methods=["POST"])
def check_label_dataset():
    """

    Returns:

    """
    label_path = request.form.get("label_data_path")
    evaluation = Evaluation()
    res = evaluation.check_label_dataset(label_path)
    if res:
        return jsonify(
            {"status": 200, "data": "The verification result is true."})
    else:
        return jsonify(
            {"status": 400, "data": "The verification result is false."})


@task.route("/execute_evaluation", methods=["POST"])
def execute_evaluation():
    """

    Returns:

    """
    task_id = request.form.get("task_id")
    label_path = request.form.get("label_path")
    chosen_evaluation = json.loads(request.form.get("chosen_evaluation"))
    evaluation = Evaluation()
    res = evaluation.evaluation_execute(task_id, label_path, chosen_evaluation)
    return jsonify(res)


@task.route("/delete_tasks", methods=["POST"])
def delete_tasks():
    """

    Returns:

    """
    task_ids = json.loads(request.form.get("task_id"))
    task_api = TaskApi()
    delete_status = task_api.delete_tasks(task_ids)
    if delete_status:
        return jsonify(
            {"status": 200, "data": "The delete result is true."})
    else:
        return jsonify(
            {"status": 400, "data": "The delete result is false."})


@task.route("/get_causal_relationship", methods=["POST"])
def get_causal_relationship():
    """

    Returns:

    """
    task_id = request.form.get("task_id")
    result_data = os.path.join(FILE_PATH, "task", task_id)
    result_dict = dict()

    task_type = TaskApi().get_task_type(task_id)
    if task_type == "1":
        dataset_path = DataSetApi.get_dataset(task_id)
        dataset_file = os.path.join(dataset_path,
                                    "node_relationship_" + task_id + ".csv")
        with open(dataset_file, "r") as res_file:
            res_list = res_file.readlines()
            result_dict.update({dataset_file: json.loads(res_list[0])})
    elif task_type == "2":
        for dir_path, _, file_names in os.walk(result_data):
            for file_name in file_names:
                result_file = os.path.join(dir_path, file_name)
                with open(result_file, "r") as res_file:
                    res_list = res_file.readlines()
                    result_dict.update({file_name: json.loads(res_list[0])})
    return jsonify(result_dict)


@task.route("/set_causal_relationship", methods=["POST"])
def set_causal_relationship():
    """

    Returns:

    """
    task_id = request.form.get("task_id")
    file_name = request.form.get("file_name")
    relationship = request.form.get("relationship")
    write_result(relationship, task_id)
    return jsonify({'status': 200, 'data': 'The save result is true.'})


@task.route("/download_file", methods=["POST"])
def download_file():
    """

    Returns:

    """
    file_name = None
    task_id = request.form.get("task_id")
    task_api = TaskApi()
    task_type = task_api.get_task_type(task_id)
    if task_type == "1":
        file_name = zip_data_file(task_id)
    elif task_type == "2":
        file_name = zip_alg_file(task_id)
    if file_name:
        response = make_response(send_file(file_name))
        response.headers["Content-Disposition"] = \
            "attachment;" \
            "filename*=UTF-8''{utf_filename}".format(
                utf_filename=(task_id + ".zip"))
        return response
    else:
        return jsonify(
            {"status": 400, "data": "The result file does not exist."})


@task.route('/get_inline_dataset_names', methods=['POST'])
def get_inline_dataset_names():
    """

    Returns:

    """
    task_id = request.form.get('task_id')
    inline_name = DataSetApi.get_inline_dataset_names()

    selected_dataset = DataSetApi.get_dataset(task_id)
    if selected_dataset:
        return jsonify({'inline_datasets': inline_name,
                        'selected_dataset': selected_dataset})
    # 没有这个task_id（一般不会）所对应的任务或者该任务的dataset为空字符串（即新
    # 任务，还未设置数据集）
    return jsonify({'inline_datasets': inline_name})


@task.route('/check_dataset', methods=['POST'])
def check_dataset():
    """

    Returns:

    """
    path = request.form.get('path')
    check_result = DataSetApi.check_dataset(path)
    if check_result:
        return jsonify({'status': 200, 'data': 'The verification result is '
                                               'true.'})
    else:
        return jsonify({'status': 403, 'data': 'The verification result is '
                                               'false.'})


@task.route('/get_algorithm_names', methods=['POST'])
def get_algorithm_names():
    """

    Returns:

    """
    task_id = request.form.get('task_id')
    selected_algorithm = AlgorithmApi.get_algorithm(task_id)

    task_type = TaskApi().get_task_type(task_id)
    default = {"1": "IID_linear_ER", "2": "PC"}
    if len(selected_algorithm) <= 0:
        selected_algorithm = {"algorithm": default[task_type]}

    if task_type == "1":
        if "IID" not in selected_algorithm['algorithm']:
            algorithm = selected_algorithm['algorithm']
            graph = "ER"
        else:
            alg_list = selected_algorithm['algorithm'].split("_")
            algorithm = "_".join(alg_list[:-1])
            graph = alg_list[-1]
        return jsonify([{"name": "样本分布", "val": algorithm,
                         "default": "IID_linear", "list": list(SEM_TYPE.keys())},
                        {"name": "因果图类型", "val": graph, "default": "ER",
                         "list": list(GRAPH_TYPE.keys())}])
    elif task_type == "2":
        algorithm_names = AlgorithmApi.get_algorithm_names()
        return jsonify([{"name": "选择算法",
                         "val": selected_algorithm['algorithm'],
                         "default": "PC",
                         "list": algorithm_names}])
    else:
        return jsonify({'status': 400, 'data': 'The task type is incorrect.'})


@task.route('/get_algorithm_parameters', methods=['POST'])
def get_algorithm_parameters():
    """
    在算法名下拉列表点击选项时触发
    :return:
    """
    task_id = request.form.get('task_id')
    selected_algorithm = request.form.get('selected_algorithm')
    selected_graph = request.form.get('selected_graph')
    algorithm_info = AlgorithmApi.get_algorithm(task_id)
    cur_params = dict()
    if len(algorithm_info) > 0:
        task_type = TaskApi().get_task_type(task_id)
        if task_type == "1":
            alg_list = algorithm_info['algorithm'].split("_")
            task_algorithm_name = "_".join(alg_list[:-1])
            task_graph_name = alg_list[-1]
        elif task_type == "2":
            task_algorithm_name = algorithm_info['algorithm']
        if selected_algorithm == task_algorithm_name:
            cur_params = algorithm_info['parameters']

    task_type = TaskApi().get_task_type(task_id)
    default_parameters = dict()
    if task_type == "1":
        default_parameters = simulate_parameters(selected_algorithm, selected_graph)
    elif task_type == "2":
        default_parameters = algorithm_parameters(selected_algorithm)

    # 选中的不是原有算法时或者该任务的算法配置为空字典（即新任务，还未设置算法）
    if not cur_params:
        cur_params = default_parameters

    res_list = list()
    for cur_key, cur_value in default_parameters.items():
        if cur_key not in cur_params.keys():
            cur_params.update({cur_key: cur_value})
        if isinstance(cur_value, bool):
            cur_params[cur_key] = str(cur_params[cur_key])
            cur_value = str(cur_value)
        param = {"name": cur_key, "val": cur_params[cur_key],
                 "default": cur_value,
                 "list": sem_type_set(cur_key, selected_algorithm)}
        res_list.append(param)
    return jsonify(res_list)


@task.route('/set_dataset_info', methods=['POST'])
def set_dataset_info():
    """

    Returns:

    """
    task_id = request.form.get('task_id')
    name_path = request.form.get('name_path')
    dataset = DataSetApi(name_path)
    result = dataset.set_dataset_info(task_id)
    if result:
        return jsonify({'status': 200, 'task_id': task_id,
                        'data': 'The dataset is set successfully.'})
    return jsonify({'status': 400, 'task_id': task_id,
                    'data': 'The dataset fail to be set.'})


@task.route('/set_algorithm_info', methods=['POST'])
def set_algorithm_info():
    """

    Returns:

    """
    task_id = request.form.get('task_id')
    algorithm = request.form.get('selected_algorithm')
    graph = request.form.get('selected_graph')
    parameters = request.form.get('selected_parameters')
    if algorithm and algorithm != "EVENT" and graph:
        name = "_".join([algorithm, graph])
    else:
        name = algorithm
    algorithm = AlgorithmApi(name, parameters)
    status_code, task_info = algorithm.set_algorithm_info(task_id)
    return jsonify({'status': status_code, 'data': task_info})


@task.route('/run_task', methods=['POST'])
def run_task():
    """

    Returns:

    """
    result = None
    task_id = request.form.get('task_id')
    dataset = DataSetApi.get_dataset(task_id)
    algorithm_info = AlgorithmApi.get_algorithm(task_id)
    tasks = {"1": run.run_data, "2": run.run_task}

    task_type = TaskApi().get_task_type(task_id)

    if dataset and len(algorithm_info) > 0 and task_type in tasks.keys():
        result = tasks[task_type](task_id, dataset,
                                  algorithm_info['algorithm'],
                                  algorithm_info['parameters'])

    if result:
        status_code = 200
        data = 'The task succeeds to begin to run.'
    else:
        status_code = 400
        data = 'The task fails to begin to run.'

    return jsonify({'status_code': status_code, 'data': data})


@task.route('/get_builtin_data_checkbox', methods=['POST'])
def get_builtin_data_checkbox():
    """

    Returns:

    """
    task_id = request.form.get('task_id')
    dataset = DataSetApi.get_dataset(task_id)
    inline_datasets = DataSetApi.get_inline_dataset_names()
    if dataset in inline_datasets:
        return jsonify({'is_inline': True})
    return jsonify({'is_inline': False})


@task.route('/get_edit_customize_dataset_path', methods=['POST'])
def get_edit_customize_dataset_path():
    """

    Returns:

    """
    task_id = request.form.get('task_id')
    dataset = DataSetApi.get_dataset(task_id)
    inline_datasets = DataSetApi.get_inline_dataset_names()
    if dataset not in inline_datasets:
        return jsonify({'customize_path': dataset})

    return jsonify({})


@task.route('/add_task', methods=['POST'])
def add_task():
    """

    Returns:

    """
    task_id = request.form.get('task_id')
    task_name = request.form.get('task_name')
    task_type = request.form.get('task_type')
    if task_id:
        task_id = TaskApi().add_task(task_type, task_name, task_id)
    else:
        # 编辑时修改任务名
        task_id = TaskApi().add_task(task_type, task_name)

    return jsonify({'task_id': task_id})
