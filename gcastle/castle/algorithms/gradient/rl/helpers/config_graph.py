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

import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


net_arg = add_argument_group('Network')
# Encoder
net_arg.add_argument('--encoder_type', type=str, default='TransformerEncoder', help='type of encoder used')
net_arg.add_argument('--hidden_dim', type=int, default=64, help='actor LSTM num_neurons')
net_arg.add_argument('--num_heads', type=int, default=16, help='actor input embedding')
net_arg.add_argument('--num_stacks', type=int, default=6, help='actor LSTM num_neurons')
net_arg.add_argument('--residual', action='store_true', help='whether to use residual for gat encoder')

# Decoder
net_arg.add_argument('--decoder_type', type=str, default='SingleLayerDecoder', help='type of decoder used')
net_arg.add_argument('--decoder_activation', type=str, default='tanh',
                     help='activation for decoder')    # Choose from: 'tanh', 'relu', 'none'
net_arg.add_argument('--decoder_hidden_dim', type=int, default=16, help='hidden dimension for decoder')
net_arg.add_argument('--use_bias', action='store_true', help='Whether to add bias term when calculating decoder logits')
net_arg.add_argument('--use_bias_constant', action='store_true', help='Whether to add bias term as CONSTANT when calculating decoder logits')
net_arg.add_argument('--bias_initial_value', type=float, default=False,
                     help='Initial value for bias term when calculating decoder logits')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batch_size', type=int, default=64, help='batch size for training')
data_arg.add_argument('--input_dimension', type=int, default=64, help='dimension of reshaped vector') # reshaped
data_arg.add_argument('--normalize', action="store_true", help='whether the inputdata shall be normalized')
data_arg.add_argument('--transpose', action="store_true", help='whether the true graph needs transposed')

data_arg.add_argument('--score_type', type=str, default='BIC', help='score functions')
data_arg.add_argument('--reg_type', type=str, default='LR', help='regressor type (in combination wth score_type)')
data_arg.add_argument('--lambda_iter_num', type=int, default=1000, help='how often to update lambdas')
#TODO: maybe add other weights adjustment strategy as an option
data_arg.add_argument('--lambda_flag_default', action="store_true",
                      help='with set lambda parameters; true with default strategy and ignore input bounds')
data_arg.add_argument('score_bd_tight', action='store_true',
                      help='if bound is tight, then simply use a fixed value, rather than the adaptive one')
data_arg.add_argument('--lambda1_update', type=float, default=1, help='increasing additive lambda1')
data_arg.add_argument('--lambda2_update', type=float, default=10, help='increasing  multiplying lambda2')
data_arg.add_argument('--score_lower', type=float, default=0.0, help='lower bound on lambda1')
data_arg.add_argument('--score_upper', type=float, default=0.0, help='upper bound on lambda1')
data_arg.add_argument('--lambda2_lower', type=float, default=-1, help='lower bound on lambda2')
data_arg.add_argument('--lambda2_upper', type=float, default=-1, help='upper bound on lambda2')

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--seed', type=int, default=8, help='seed')
train_arg.add_argument('--nb_epoch', type=int, default=20000, help='nb epoch')
train_arg.add_argument('--lr1_start', type=float, default=0.001, help='actor learning rate')
train_arg.add_argument('--lr1_decay_step', type=int, default=5000, help='lr1 decay step')
train_arg.add_argument('--lr1_decay_rate', type=float, default=0.96, help='lr1 decay rate')
train_arg.add_argument('--alpha', type=float, default=0.99, help='update factor moving average baseline')
train_arg.add_argument('--init_baseline', type=float, default=-1.0, help='initial baseline - REINFORCE')
train_arg.add_argument('--temperature', type=float, default=3.0, help='pointer_net initial temperature')
train_arg.add_argument('--C', type=float, default=10.0, help='pointer_net tan clipping')
train_arg.add_argument('--l1_graph_reg', type=float, default=0.0, help='L1 graph regularization to encourage sparsity')

# Misc
misc_arg = add_argument_group('User options') #####################################################
misc_arg.add_argument('--inference_mode', type=str2bool, default=True, help='switch to inference mode when model is trained') 
misc_arg.add_argument('--verbose', type=bool, default=False, help='print detailed logging or not')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
