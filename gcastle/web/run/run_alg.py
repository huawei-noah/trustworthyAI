import numpy as np
import datetime
import json
from loguru import logger

from castle.algorithms.pc.pc import PC
from castle.algorithms.gradient.graph_auto_encoder import GAE
from castle.algorithms.gradient.gran_dag import GraN_DAG, Parameters
from castle.algorithms.gradient.notears import NotearsLowRank
from castle.algorithms.gradient.notears import NotearsMLP
from castle.algorithms.gradient.notears import NotearsSob
from castle.algorithms.ttpm import TTPM
from castle.algorithms.gradient.notears.linear import Notears
from castle.common.base import Tensor, EventTensor
from castle.common.plot_dag import GraphDAG
from castle.datasets import IIDSimulation, EventSimulation

from web.models.task_db import TaskApi
from web.common.utils import translation_parameters, write_result


def run_pc(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    if data is None:
        dataset = IIDSimulation(n=1000, d=10, s=10, graph_type='ER',
                                method='linear',
                                sem_type='gauss', seed=123)
        true_dag, sample = dataset.W, dataset.X
        data = Tensor(sample)
    elif isinstance(data, tuple):
        true_dag, data, _ = data
    parameters = translation_parameters(parameters)
    model = PC()
    model.learn(data, **parameters)
    return model, true_dag


def run_gae(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    if data is None:
        dataset = IIDSimulation(n=2000, d=20, degree=5,
                                graph_type='hierarchical',
                                method='nonlinear', sem_type='quadratic',
                                graph_level=10, seed=123)
        true_dag, sample = dataset.W, dataset.X
        data = Tensor(sample)
    elif isinstance(data, tuple):
        true_dag, data, _ = data
    parameters = translation_parameters(parameters)
    model = GAE()
    model.learn(data, **parameters)
    return model, true_dag


def run_gran(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    parameters = translation_parameters(parameters)
    if data is None:
        dataset = IIDSimulation(n=1000, d=10, s=10, graph_type='ER',
                                method='nonlinear',
                                sem_type='mlp', seed=123)
        true_dag, data = dataset.W, dataset.X
        data = Tensor(data)
        parameters = Parameters(input_dim=data.data.shape[1])
    elif isinstance(data, tuple):
        true_dag, data, _ = data
        parameters = Parameters(input_dim=data.data.shape[1])

    # Instantiation algorithm
    model = GraN_DAG(params=parameters)
    model.learn(data=data, target=true_dag)
    return model, true_dag


def run_low_rank(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    if data is None:
        dataset = IIDSimulation(n=1000, d=10, degree=3, rank=5,
                                graph_type='LR',
                                method='linear', sem_type='gauss', seed=123)
        true_dag, sample = dataset.W, dataset.X

        # calculate dag rank
        rank = np.linalg.matrix_rank(true_dag)
        # notears-low-rank learn
        data = Tensor(sample)
    elif isinstance(data, tuple):
        true_dag, data, _ = data
        rank = np.linalg.matrix_rank(true_dag)
    model = NotearsLowRank()
    model.learn(data, rank=rank)
    return model, true_dag


def run_mlp(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    if data is None:
        dataset = IIDSimulation(n=1000, d=10, s=10, graph_type='ER',
                                method='nonlinear',
                                sem_type='mlp', seed=123)
        true_dag, sample = dataset.W, dataset.X

        # notears-mlp learn
        data = Tensor(sample)
    elif isinstance(data, tuple):
        true_dag, data, _ = data
    model = NotearsMLP()
    model.learn(data)
    return model, true_dag


def run_notears(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    if data is None:
        dataset = IIDSimulation(n=1000, d=10, s=10, graph_type='ER',
                                method='linear',
                                sem_type='gauss', seed=123)
        true_dag, sample = dataset.W, dataset.X

        # notears learn
        data = Tensor(sample)
    elif isinstance(data, tuple):
        true_dag, data, _ = data
    parameters = translation_parameters(parameters)
    model = Notears()
    model.learn(data, **parameters)
    return model, true_dag


def run_sob(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    if data is None:
        dataset = IIDSimulation(n=1000, d=10, s=10, graph_type='ER',
                                method='nonlinear',
                                sem_type='mlp', seed=123)
        true_dag, sample = dataset.W, dataset.X

        # notears-sob learn
        data = Tensor(sample)
    elif isinstance(data, tuple):
        true_dag, data, _ = data
    model = NotearsSob()
    model.learn(data)
    return model, true_dag


def run_ttpm(data, parameters):
    """

    Parameters
    ----------
    data
    parameters

    Returns
    -------

    """
    true_dag = None
    topo = None
    parameters = translation_parameters(parameters)
    if data is None:
        dataset = EventSimulation(n=20, NE_num=40, sample_size=5000, seed=123)
        event_table, topo, true_dag = dataset.event_table, dataset.topo, dataset.edge_mat
        data = EventTensor(event_table)
    elif isinstance(data, tuple):
        true_dag, data, topo = data

    model = TTPM()
    model.learn(tensor=data, topo=topo, **parameters)
    return model, true_dag


def read_data(data, kwargs):
    """

    Parameters
    ----------
    data
    kwargs

    Returns
    -------

    """
    if data is not None:
        if isinstance(data, str):
            if 'header' in kwargs and isinstance(kwargs['header'], int):
                data = Tensor(data, header=int(kwargs['header']))
            elif 'header' in kwargs:
                print('Error: header is not a number!')
            else:
                data = Tensor(data)
    return data


def run(data, alg='pc', task_id=None, **kwargs):
    """
    必须参数：
        data: 文件路径/二维列表/numpy二维数组/pandas数据框
        alg: pc(待扩充)
    可选参数（kwargs）：
        header: 表头是否存在，如果第一行为表头，则值为 0
        labels: 最终DAG节点的名称，默认用数字序号
    """
    task_api = TaskApi()
    try:
        start_time = datetime.datetime.now()
        # import data
        data = read_data(data, kwargs)
        task_api.update_performance(task_id, "", dict())
        task_api.update_task_status(task_id, 0.1)
        task_api.update_consumed_time(task_id, start_time)
        task_api.update_update_time(task_id, start_time)
        # algorithm
        run_dict = {"pc": run_pc, "gae": run_gae, "gran": run_gran,
                    "low_rank": run_low_rank, "mlp": run_mlp,
                    "notears": run_notears, "sob": run_sob, "ttpm": run_ttpm}

        p_res, true_dag = run_dict[alg](data, kwargs['parameters'])
        task_api.update_task_status(task_id, 0.5)
        task_api.update_consumed_time(task_id, start_time)

        print("p.causal_matrix.values", p_res.causal_matrix.values)
        task_api.update_est_dag(task_id, p_res.causal_matrix.values)
        task_api.update_true_dag(task_id, true_dag)

        # deal result
        gragh = GraphDAG.nx_graph(p_res.causal_matrix)
        task_api.update_task_status(task_id, 0.8)
        task_api.update_consumed_time(task_id, start_time)

        diagrams = list(gragh.edges)
        edges = [[str(diagram[0]), str(diagram[1])] for diagram in diagrams if
         isinstance(diagram, tuple) and len(diagram) > 1]
        write_result(json.dumps(edges), task_id)
        task_api.update_task_status(task_id, 1.0)
        task_api.update_consumed_time(task_id, start_time)
    except Exception as error:
        task_api.update_task_status(task_id, str(error))
        task_api.update_consumed_time(task_id, start_time)
        logger.warning('alg run fail %s' % str(error))


if __name__ == '__main__':
    path = r"D:\homework\homework\PCastle_debug\web\data\gmD_discrete.csv"
    algorithm_list = ["pc", "gae", "gran", "low_rank", "mlp", "notears", "sob",
                      "ttpm"]
    algorithm = algorithm_list[7]
    task_id = algorithm_list[7]
    run(data="", alg=algorithm.lower(), task_id=task_id, kwargs={"header": 0})
