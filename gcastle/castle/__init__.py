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

__version__ = "1.0.3"


import sys
if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')
import os
import yaml
import logging
import pandas as pd

from castle.common import GraphDAG
from castle.metrics import MetricsDAG


logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


def _import_algo(algo):
    """
    import algorithm corresponding to `algo`

    Parameters
    ----------
    algo: str
        lowercase letters of the algorithm `algo`

    Returns
    -------
    out: class object
        castle algorithm
    """

    if algo.lower() == 'pc':
        from castle.algorithms import PC as Algorithm
    elif algo.lower() == 'anm':
        from castle.algorithms import ANMNonlinear as Algorithm
    elif algo.lower() == 'icalingam':
        from castle.algorithms import ICALiNGAM as Algorithm
    elif algo.lower() == 'directlingam':
        from castle.algorithms import DirectLiNGAM as Algorithm
    elif algo.lower() == 'notears':
        from castle.algorithms import Notears as Algorithm
    elif algo.lower() == 'notearslowrank':
        from castle.algorithms import NotearsLowRank as Algorithm
    elif algo.lower() == 'notearsnonlinear':
        from castle.algorithms import NotearsNonlinear as Algorithm
    elif algo.lower() == 'corl':
        from castle.algorithms import CORL as Algorithm
    elif algo.lower() == 'rl':
        from castle.algorithms import RL as Algorithm
    elif algo.lower() == 'gae':
        from castle.algorithms import GAE as Algorithm
    elif algo.lower() == 'ges':
        from castle.algorithms import GES as Algorithm
    elif algo.lower() == 'golem':
        from castle.algorithms import GOLEM as Algorithm
    elif algo.lower() == 'grandag':
        from castle.algorithms import GraNDAG as Algorithm
    elif algo.lower() == 'pnl':
        from castle.algorithms import PNL as Algorithm
    else:
        raise ValueError('Unknown algorithm.==========')

    logging.info(f"import algorithm corresponding to {algo} complete!")

    return Algorithm


def fast_start(algo_name, yaml_path, x=None, true_dag=None, plot=False, save_dir=None, **kwargs):
    """
    call this function to run castle algorithm

    Parameters
    ----------
    algo_name: str
        lowercase letters of the algorithm to run. e.g. 'pc'
    yaml_path: str
        path of `.yaml`. you should copy yaml file from castle.params_config_template.yaml to
        another directory.
    x: array-like
        IID data, numpy.array.
    true_dag: array-like
        true causal matrix, numpy.array.
    plot: bool
        whether show plot of estimate causal matrix.
    save_dir: str
        directory path to save result, if None, save result to current directory.
    kwargs:
        these keyword arguments is for `learn` method for some algorithms.

    Notes
    -----
    If x is not None, x will take effect, otherwise, the attribute dataset in yaml file will be read.
    In the same way.
    If true_dag is not None, y will take effect, otherwise, the attribute dataset in yaml file will be read.

    Examples
    --------
    >>> # if you want to run PC for your data, you can use this method like the following steps
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> from castle.datasets import load_dataset
    >>>
    >>> # you can also use your own real data
    >>> X, true_dag, _ = load_dataset(name='IID_Test')
    >>>
    >>> pred_dag = fast_start('pc', x, yaml_config='params_config_template.yaml',
    >>>                       true_dag=true_dag, plot=True)
    >>>
    >>> # Evaluation model
    >>> met = MetricsDAG(pred_dag, true_dag)
    >>> print(met.metrics)
    {'fdr': 0.0667, 'tpr': 0.7778, 'fpr': 0.037, 'shd': 4, 'nnz': 15, 'precision': 0.6667,
    'recall': 0.7778, 'F1': 0.7179, 'gscore': 0.3889}

    Returns
    -------
    out: array-like
        causal matrix
    """

    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise e

    if x is None:
        x = pd.read_csv(config['dataset']['X'], dtype=float).values
    if true_dag is None and config['dataset']['true_dag'] is not None:
        true_dag = pd.read_csv(config['dataset']['true_dag'],
                               dtype=int,
                               index_col=0,
                               header=0).values

    logging.info(f"start running {algo_name} algorithm...==========")
    Algorithm = _import_algo(algo_name)
    algo = Algorithm(**config[algo_name])
    algo.learn(data=x, **kwargs)
    pre_dag = algo.causal_matrix

    if save_dir is None:
        save_dir = config['save']['directory']
    if save_dir is not None:
        df = pd.DataFrame(pre_dag)
        save_file = os.path.join(save_dir, f"{algo_name}_result.csv")
        df.to_csv(save_file, index=False, encoding='utf-8')
        logging.info(f"save result complete! Directory: {save_file}==========")

    if true_dag is not None:
        met = MetricsDAG(pre_dag, true_dag)
        logging.info(f"metrics of causal matrix: ==========\n {met.metrics}")

    if plot:
        GraphDAG(pre_dag, true_dag)

    return pre_dag
