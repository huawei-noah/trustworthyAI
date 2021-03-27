# coding=utf-8
# 2021.03 deleted (1) yaml; 
#                 (2) load_yaml_config, save_yaml_config; 
#                 (3) add_dataset_args
# Huawei Technologies Co., Ltd. 
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Copyright (c) Ignavier Ng (https://github.com/ignavier/golem)
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

import argparse
import sys


def get_args():
    
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    add_training_args(parser)
    add_other_args(parser)

    args, unparsed = parser.parse_known_args()
    return args, unparsed


def add_model_args(parser):
    """
    Add model arguments for parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Parser.
    """
    parser.add_argument('--B_init',
                        default=None,
                        help="File of weighted matrix for initialization. Set to None to disable.")

    parser.add_argument('--lambda_1',
                        type=float,
                        default=0.0,
                        help="Coefficient of L1 penalty.")

    parser.add_argument('--lambda_2',
                        type=float,
                        default=0.0,
                        help="Coefficient of DAG penalty.")

    parser.add_argument('--equal_variances',
                        dest='equal_variances',
                        action='store_true',
                        help="Assume equal noise variances for likelibood objective.")

    parser.add_argument('--non_equal_variances',
                        dest='equal_variances',
                        action='store_false',
                        help="Assume non-equal noise variances for likelibood objective.")


def add_training_args(parser):
    """
    Add training arguments for parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Parser.
    """
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help="Learning rate of Adam optimizer.")

    parser.add_argument('--num_iter',
                        type=int,
                        default=1e+5,
                        help="Number of iterations for training.")

    parser.add_argument('--checkpoint_iter',
                        type=int,
                        default=5000,
                        help="Number of iterations between each checkpoint. Set to None to disable.")


def add_other_args(parser):
    """
    Add other arguments for parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
        Parser.
    """
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="Random seed.")

    parser.add_argument('--graph_thres',
                        type=float,
                        default=0.3,
                        help="Threshold for weighted matrix.")
