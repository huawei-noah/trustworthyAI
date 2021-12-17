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


import torch
import yaml
import argparse
import logging
import pandas as pd
from castle.datasets.loader import load_dataset
from example.example import read_file, run_simulate, save_to_file, train

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.ERROR)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generic runner')
    parser.add_argument('-c', '--config',
                        dest='config',
                        help='You must provide the .yaml file corresponding to '
                             'the simulation data or algorithm.',
                        default='./example/dataset/simulate_data.yaml')
    parser.add_argument('-p', '--plot',
                        dest='plot',
                        help='whether show graph.',
                        default=True)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        try:
            params_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if params_config['task_params']['task_type'] == 1:
        logging.info('Start task 1: simulate dataset.')
        outer = run_simulate(config=params_config)
        save_to_file(outer[0],
                     file=params_config['dataset_params']['x_file'])
        logging.info(f"Dataset X has been saved to "
                     f"{params_config['dataset_params']['x_file']}.")
        save_to_file(outer[1],
                     file=params_config['dataset_params']['dag_file'])
        logging.info(f"Dataset true_dag has been saved to "
                     f"{params_config['dataset_params']['dag_file']}.")
        if params_config['task_params']['algorithm'] == 'EVENT':
            save_to_file(outer[2],
                         file=params_config['dataset_params']['topology_file'])
            logging.info(f"Dataset topology has been saved to "
                         f"{params_config['dataset_params']['topology_file']}.")
        logging.info('Task completed!')
    elif params_config['task_params']['task_type'] == 2:
        logging.info('Start task 2: causal discovery.')

        topology = None
        if params_config['dataset_params']['x_file'] is None:

            if params_config['task_params']['algorithm'] == 'TTPM':
                X, true_dag, topology = load_dataset('THP_Test')
            else:
                X, true_dag, _ = load_dataset('IID_Test')
                X = pd.DataFrame(X)
        else:
            X = read_file(file=params_config['dataset_params']['x_file'],
                          header=0)
            true_dag = read_file(file=params_config['dataset_params']['dag_file'],
                                 header=None)

        if 'topology_file' in params_config['dataset_params'].keys():
            topology = read_file(file=params_config['dataset_params']['topology_file'],
                                 header=None)

        train(model_name=params_config['task_params']['algorithm'],
              X=X,
              true_dag=true_dag,
              model_params=params_config['algorithm_params'],
              topology_matrix=topology,
              plot=args.plot)
    else:
        raise ValueError('Invalid value of the configuration parameter '
                         'task_type, expected integer 1 or 2.')
