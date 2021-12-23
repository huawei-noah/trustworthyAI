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

import json
import os
import yaml
from flask import Blueprint, make_response, send_file, request, jsonify

import castle
from example.example import read_file

from web.metrics.evaluation import Evaluation
from web.models.base_class import DataSetApi, AlgorithmApi
from web.models.task_db import TaskApi
from web.run import run
from web.common.config import SEM_TYPE, FILE_PATH, INLINE_DATASETS
from web.common.utils import algorithm_parameters, zip_alg_file, zip_data_file, \
    sem_type_set, write_result, conversion_type, save_gragh_edges, set_current_language, _, update_inline_datasets
from web import __version__


task = Blueprint("task", __name__)


@task.route("/get_task_list", methods=["GET"])
def get_task_list():
    """
    @@@
    ### description
    > Back to Task List.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|

    ### request
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_api = TaskApi()
    tasks = task_api.list_tasks()
    return jsonify(tasks)


@task.route("/get_label_checkbox", methods=["POST"])
def get_label_checkbox():
    """
    @@@
    ### description
    > Checking Whether Data Is Built-in Data
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    is_inline = evaluation.get_label_checkbox(int(task_id))
    return jsonify({"is_inline": str(is_inline)})

@task.route("/get_evaluation_results", methods=["POST"])
def get_evaluation_results():
    """
    @@@
    ### description
    >
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    task_api = TaskApi()
    performance = task_api.get_performance(task_id)
    return jsonify(performance)


@task.route("/get_evaluation_metrics", methods=["POST"])
def get_evaluation_metrics():
    """
    @@@
    ### description
    > Obtain all evaluation indicators.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    return jsonify(evaluation.get_evaluation_metrics(int(task_id)))


@task.route("/get_builtin_label", methods=["POST"])
def get_builtin_label():
    """
    @@@
    ### description
    > Obtains the built-in data name.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    return jsonify(evaluation.get_builtin_label(int(task_id)))


@task.route("/get_customize_label", methods=["POST"])
def get_customize_label():
    """
    @@@
    ### description
    > Obtain the path of the customized data file.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    evaluation = Evaluation()
    return jsonify(evaluation.get_task_customize_label(int(task_id)))


@task.route("/check_label_dataset", methods=["POST"])
def check_label_dataset():
    """
    @@@
    ### description
    > Check whether the real image exists.
    ### args
    |      args     | nullable | request type | type |        remarks           |
    |---------------|----------|--------------|------|--------------------------|
    |label_data_path|  false   |    body      | str  |Checking Realistic Path   |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    label_path = request.form.get("label_data_path")
    evaluation = Evaluation()
    res = evaluation.check_label_dataset(label_path)
    if res:
        return jsonify(
            {"status": 200, "data": _("The verification result is true.")})
    else:
        return jsonify(
            {"status": 400, "data": _("The verification result is false.")})


@task.route("/execute_evaluation", methods=["POST"])
def execute_evaluation():
    """
    @@@
    ### description
    > Perform algorithm evaluation.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |
    |  label_path  |  false   |    body      | str  | Realistic map |
    |chosen_evaluation|  false   |    body   | str  | Evaluation Indicator List |

    ### request
    ```json
    {"task_id": 1, "label_path":xxxx, "chosen_evaluation":["F1", "fdr"]}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
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
    @@@
    ### description
    > Deleting Tasks.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | str  | task list |

    ### request
    ```json
    {"task_id": [1,3,4]}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_ids = json.loads(request.form.get("task_id"))
    task_api = TaskApi()
    delete_status = task_api.delete_tasks(task_ids)
    if delete_status:
        return jsonify(
            {"status": 200, "data": _("The delete result is true.")})
    else:
        return jsonify(
            {"status": 400, "data": _("The delete result is false.")})


@task.route("/get_causal_relationship", methods=["POST"])
def get_causal_relationship():
    """
    @@@
    ### description
    > Obtains the causal relationship of a task.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    result_data = os.path.join(FILE_PATH, "task", task_id)
    result_dict = dict()

    task_type = TaskApi().get_task_type(task_id)
    if task_type == 1:
        dataset_path = DataSetApi.get_dataset(task_id)
        task_name = TaskApi().get_task_name(task_id)
        dataset_file = os.path.join(dataset_path,
                                    "node_relationship_" + str(task_id) + "_" + task_name + ".csv")
        with open(dataset_file, "r") as res_file:
            res_list = res_file.readlines()
            result_dict.update({dataset_file: json.loads(res_list[0])})
    elif task_type == 2:
        for dir_path, _, file_names in os.walk(result_data):
            for file_name in file_names:
                result_file = os.path.join(dir_path, file_name)
                with open(result_file, "r") as res_file:
                    res_list = res_file.readlines()
                    result_dict.update({file_name: json.loads(res_list[0])})
        if len(result_dict) > 1:
            pre_list = [tuple(pl) for pl in list(result_dict.values())[0]]
            true_list = [tuple(tl) for tl in list(result_dict.values())[1]]
            result_dict["common"] = list(set(pre_list).intersection(set(true_list)))
            result_dict["pre_common"] = list(set(pre_list).difference(set(true_list)))
            result_dict["true_common"] = list(set(true_list).difference(set(pre_list)))
    return jsonify(result_dict)


@task.route("/set_causal_relationship", methods=["POST"])
def set_causal_relationship():
    """
    @@@
    ### description
    > Modifying the Causality Diagram.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |
    | relationship |  false   |    body      | str  | Modified Causality |
    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    relationship = request.form.get("relationship")
    write_result(relationship, task_id)
    return jsonify({'status': 200, 'data': _('The save result is true.')})


@task.route("/download_file", methods=["POST"])
def download_file():
    """
    @@@
    ### description
    > Downloading task-related files in the list.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    file_name = None
    task_id = request.form.get("task_id")
    task_api = TaskApi()
    task_type = task_api.get_task_type(task_id)
    if task_type == 1:
        data_path = DataSetApi.get_dataset(task_id)
        if data_path is None:
            data_path = os.path.join(FILE_PATH, 'inline')

        task_api = TaskApi()
        task_name = task_api.get_task_name(task_id)
        file_name = zip_data_file(task_id, task_name, data_path)
    elif task_type == 2:
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
            {"status": 400, "data": _("The result file does not exist.")})


@task.route('/get_inline_dataset_names', methods=['POST'])
def get_inline_dataset_names():
    """
    @@@
    ### description
    > Obtains the built-in data name list.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 2}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get('task_id')
    update_inline_datasets()
    inline_name = INLINE_DATASETS
    selected_dataset = DataSetApi.get_dataset(task_id)
    if selected_dataset:
        return jsonify({'inline_datasets': inline_name,
                        'selected_dataset': selected_dataset})
    return jsonify({'inline_datasets': inline_name})


@task.route('/check_dataset', methods=['POST'])
def check_dataset():
    """
    @@@
    ### description
    > Check whether the file path exists.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    path      |  false   |    body      | str  | file path string  |

    ### request
    ```json
    {"path": xxxx}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    path = request.form.get('path')
    check_result = DataSetApi.check_dataset(path)

    if check_result:
        return jsonify({"column_num": check_result})
    else:
        return jsonify({'status': 403, 'data': _('The verification result is false.')})


@task.route('/get_algorithm_names', methods=['POST'])
def get_algorithm_names():
    """
    @@@
    ### description
    > Obtains the algorithm name list.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get('task_id')
    selected_algorithm = AlgorithmApi.get_algorithm(task_id)

    task_type = TaskApi().get_task_type(task_id)
    default = {1: "IID_LINEAR", 2: "PC"}

    if len(selected_algorithm) <= 0:
        selected_algorithm = {"algorithm": default[task_type]}

    if task_type == 1:
        return jsonify({"name": _("Sample Distribution"), "val": selected_algorithm['algorithm'],
                         "default": default[task_type], "list": list(SEM_TYPE.keys())})
    elif task_type == 2:
        algorithm_names = AlgorithmApi.get_algorithm_names()
        return jsonify({"name": _("Selecting an algorithm"),
                         "val": selected_algorithm['algorithm'],
                         "default": default[task_type],
                         "list": algorithm_names})
    else:
        return jsonify({'status': 400, 'data': _('The task type is incorrect.')})


@task.route('/get_algorithm_parameters', methods=['POST'])
def get_algorithm_parameters():
    """
    @@@
    ### description
    > Obtains algorithm parameters. This function is triggered when an option is selected from the Algorithm Name drop-down list.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |
    |selected_algorithm|  false   |    body      | str  | algorithm string |
    ### request
    ```json
    {"task_id": 1, "selected_algorithm": "PC"}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """

    task_id = request.form.get('task_id')
    selected_algorithm = request.form.get('selected_algorithm')
    algorithm_info = AlgorithmApi.get_algorithm(task_id)
    cur_params = dict()
    if len(algorithm_info) > 0:
        task_algorithm_name = algorithm_info['algorithm']
        if selected_algorithm == task_algorithm_name:
            cur_params = algorithm_info['parameters']
    default_parameters = algorithm_parameters(selected_algorithm)

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
    @@@
    ### description
    > Setting Task Data Information.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |
    |   name_path  |  false   |    body      | str  | task path string         |
    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get('task_id')
    name_path = request.form.get('name_path')
    dataset = DataSetApi(name_path)
    result = dataset.set_dataset_info(task_id)
    if result:
        return jsonify({'status': 200, 'task_id': task_id,
                        'data': _('The dataset is set successfully.')})
    return jsonify({'status': 400, 'task_id': task_id,
                    'data': _('The dataset fail to be set.')})


@task.route('/set_algorithm_info', methods=['POST'])
def set_algorithm_info():
    """
    @@@
    ### description
    > Set the operators and parameters of the causal discovery or data generation task.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |
    |selected_algorithm|  false   |    body  | str  | algorithm string         |
    |selected_parameters|  false   |    body      | str  | Dictionary Serialization String |
    ### request
    ```json
    {"task_id": 1, "selected_algorithm": "PC", "selected_parameters":dict}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get('task_id')
    algorithm_name = request.form.get('selected_algorithm')
    parameters = request.form.get('selected_parameters')
    if algorithm_name == 'IID_NONLINEAR' or algorithm_name == 'IID_LINEAR':
        parameters = json.loads(parameters)
        parameters.update({'method': algorithm_name.split("_")[-1].lower()})
        parameters = json.dumps(parameters)
    algorithm = AlgorithmApi(algorithm_name, parameters)
    status_code, task_info = algorithm.set_algorithm_info(task_id)
    return jsonify({'status': status_code, 'data': task_info})


@task.route('/run_task', methods=['POST'])
def run_task():
    """
    @@@
    ### description
    > Perform causal discovery or data generation tasks.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    result = None
    task_id = request.form.get('task_id')
    dataset = DataSetApi.get_dataset(task_id)
    algorithm_info = AlgorithmApi.get_algorithm(task_id)
    tasks = {1: run.run_data, 2: run.run_task}

    task_type = TaskApi().get_task_type(task_id)

    if dataset and len(algorithm_info) > 0 and task_type in tasks.keys():
        result = tasks[task_type](task_id, dataset,
                                  algorithm_info['algorithm'],
                                  algorithm_info['parameters'])

    if result:
        status_code = 200
        data = _('The task succeeds to begin to run.')
    else:
        status_code = 400
        data = _('The task fails to begin to run.')

    return jsonify({'status_code': status_code, 'data': data})


@task.route('/get_builtin_data_checkbox', methods=['POST'])
def get_builtin_data_checkbox():
    """
    @@@
    ### description
    > Obtains the current data type.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
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
    @@@
    ### description
    > Obtaining a User-Defined Data Path.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
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
    @@@
    ### description
    > New Task.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |
    |    task_name |  false   |    body      | str  | task name                |
    |    task_type |  false   |    body      | str  | task type                |
    ### request
    ```json
    {"task_id": 1, "task_name": xxxx, "task_type": 2}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get('task_id')
    task_name = request.form.get('task_name')
    task_type = int(request.form.get('task_type'))

    if task_id:
        task_id = TaskApi().add_task(task_type, task_name, task_id)
        dataset = DataSetApi.get_dataset(task_id)
        if dataset == "":
            dataset = os.path.join(FILE_PATH, 'inline')
    else:
        task_id = TaskApi().add_task(task_type, task_name)

        if task_type == 1:
            dataset = os.path.join(FILE_PATH, 'inline')
        else:
            update_inline_datasets()
            if INLINE_DATASETS:
                dataset = INLINE_DATASETS[0]
            else:
                dataset = None
    return jsonify({'task_id': task_id, 'task_path': dataset})


@task.route('/upload_config_file', methods=['POST'])
def upload_config_file():
    """
    @@@
    ### description
    > Uploading Configuration Files.
    ### files
    |      files   | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    file      |  false   |    body      | str  | file name string         |

    ### request
    ```json
    {"file": xxxx}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    f = request.files['file']
    file_name = os.path.join(FILE_PATH, f.filename)
    f.save(file_name)
    task_id = read_task(file_name)
    new_file_name = os.path.join(FILE_PATH, str(task_id) + "_" + f.filename)
    if os.path.exists(new_file_name):
        os.remove(new_file_name)
    os.rename(file_name, new_file_name)
    status_code = 200
    data = _('File uploaded successfully.')

    return jsonify({'status': status_code, 'data': data})


@task.route("/export_task", methods=["POST"])
def export_task():
    """
    @@@
    ### description
    > Interface for Exporting Configuration Files.
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    task_id = request.form.get("task_id")
    file_name = save_param(task_id)
    task_name = TaskApi().get_task_name(task_id)
    if file_name:
        response = make_response(send_file(file_name))
        response.headers["Content-Disposition"] = \
            "attachment;" \
            "filename*=UTF-8''{utf_filename}".format(
                utf_filename=(task_name + ".yaml"))
        return response
    else:
        return jsonify(
            {"status": 400, "data": _("The configuration file does not exist.")})


@task.route("/current_language", methods=["GET", "POST"])
def current_language():
    """
    @@@
    ### description
    >Switching the GUI Language
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    language = request.form.get("language")
    res = set_current_language(language)
    if res:
        status_code = 200
        data = _('The language is set successfully.')
    else:
        status_code = 400
        data = _('Failed to set the language.')

    return jsonify({'status': status_code, 'data': data})


@task.route("/get_version", methods=["GET", "POST"])
def get_version():
    """
    @@@
    ### description
    > Obtaining the Version Number
    ### args
    |      args    | nullable | request type | type |        remarks           |
    |--------------|----------|--------------|------|--------------------------|
    |    task_id   |  false   |    body      | int  | task key in the database |

    ### request
    ```json
    {"task_id": 1}
    ```
    ### return
    ```json
    {"status": xxxx, "data": xxxx}
    ```
    @@@
    """
    return jsonify({"web_version": __version__,
                    "gcastle_version": castle.__version__})


def read_task(file_name):
    """
    Read the configuration file and generate the task.

    Parameters
    ----------
    file_name:str
        profile path.

    Returns
    -------
    task_id: int
        task key in the database.
    """
    with open(file_name, 'r') as file:
        try:
            params_config = yaml.safe_load(file)
            if "task_params" in params_config.keys():
                alg = params_config['task_params']['algorithm']
                task_name = params_config['task_params']['task_name']
                task_type = params_config['task_params']['task_type']
            else:
                alg = file_name.split(os.path.sep)[-1].split(".")[0].upper()
                task_name = alg
                task_type = 2
            task_id = TaskApi().add_task(task_type, task_name)

            if "path" in params_config['dataset_params'].keys():
                x_file = params_config['dataset_params']['path']
                dag_file = None
            else:
                x_file = params_config['dataset_params']['x_file']
                dag_file = params_config['dataset_params']['dag_file']

            if x_file is not None:
                dataset = DataSetApi(x_file)
                dataset.set_dataset_info(task_id)
            if dag_file is not None:
                true_dag = read_file(dag_file)
                TaskApi().update_true_dag(task_id, true_dag)
                task_path = os.path.join(FILE_PATH, 'task', task_id)
                file_name = os.path.join(task_path, "true.txt")
                save_gragh_edges(true_dag, file_name)
            algorithm = AlgorithmApi(alg, json.dumps(params_config['algorithm_params']))
            algorithm.set_algorithm_info(task_id)
        except yaml.YAMLError as exc:
            print(exc)
    return task_id


def save_param(task_id):
    """
    Save the exported configuration file.

    Parameters
    ----------
    task_id: int
        task key in the database.

    Returns
    -------
    filename: str
        profile path.
    """
    alg = AlgorithmApi.get_algorithm(task_id)
    alg_name = None
    if 'algorithm' in alg.keys():
        alg_name = alg['algorithm']

    alg_params = None
    if 'parameters' in alg.keys():
        alg_params = alg['parameters']

    path = DataSetApi.get_dataset(task_id)
    task_type = TaskApi().get_task_type(task_id)
    task_name = TaskApi().get_task_name(task_id)
    task_data = {
        "algorithm_params": conversion_type(alg_name, alg_params),
        "task_params": {
            "algorithm": alg_name,
            "task_type": task_type,
            "task_name": task_name}
    }

    if task_type == 2:
        if path in DataSetApi.get_inline_dataset_names():
            path_task_id = path.split("_")[0]
            sample_path = os.path.join(FILE_PATH, "sample_" + path_task_id + ".csv")
            true_dag_path = os.path.join(FILE_PATH, "true_dag_" + path_task_id + ".npz")
            task_data.update({"dataset_params": {"path": path,
                                                 "x_file": sample_path,
                                                 "dag_file": true_dag_path}})
            if alg_name == "TTPM":
                topo_path = os.path.join(FILE_PATH, "topo_" + path_task_id + ".npz")
                task_data["dataset_params"].update({"topology_file": topo_path})
        else:
            task_data.update({"dataset_params": {"x_file": path,
                                                 "dag_file": None}})
            if alg_name == "TTPM":
                task_data["dataset_params"].update({"topology_file": None})
    else:
        sample_path = os.path.join(path, "sample_" + task_id + ".csv")
        true_dag_path = os.path.join(path, "true_dag_" + task_id + ".npz")
        task_data.update({"dataset_params": {"path": path,
                                             "x_file": sample_path,
                                             "dag_file": true_dag_path}})
        if alg_name == "EVENT":
            topo_path = os.path.join(path, "topo_" + task_id + ".npz")
            task_data["dataset_params"].update({"topology_file": topo_path})

    filename = os.path.join(FILE_PATH, task_name + ".yaml")
    with open(filename, 'w') as dumpfile:
        try:
            dumpfile.write(yaml.dump(task_data))
        except yaml.YAMLError as exc:
            print(exc)
    return filename
