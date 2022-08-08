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


LOG_FREQUENCY = 200

LOG_FORMAT = ("[%(asctime)s][%(filename)s - line %(lineno)s] "
              "- %(levelname)s - %(message)s")


# ==========================================================
# const variable used to check arguments for each algorithm

# key of dict denotes argument name of function;
# corresponding value denotes valid value for key.
# ==========================================================

# CORL
CORL_VALID_PARAMS = {
    'encoder_name': ['transformer', 'lstm', 'mlp'],
    'decoder_name': ['lstm', 'mlp'],
    'reward_mode': ['episodic', 'dense'],
    'reward_score_type': ['BIC', 'BIC_different_var'],
    'reward_regression_type': ['LR']
}

# RL
RL_VALID_PARAMS = {
    'encoder_type': ['TransformerEncoder', 'GATEncoder'],
    'decoder_type': ['SingleLayerDecoder', 'TransformerDecoder',
                     'BilinearDecoder', 'NTNDecoder'],
    'decoder_activation': ['tanh', 'relu', 'none'],
    'score_type': ['BIC', 'BIC_different_var'],
    'reg_type': ['LR', 'QR']
}

# GraNDAG
GRANDAG_VALID_PARAMS = {
    'model_name': ['NonLinGaussANM', 'NonLinGauss'],
    'nonlinear': ['leaky-relu', 'sigmoid'],
    'optimizer': ['rmsprop', 'sgd'],
    'norm_prod': ['paths', 'none']
}

# Notears
NOTEARS_VALID_PARAMS = {
    'loss_type': ['l2', 'logistic', 'poisson']
}

# nonlinear Notears
NONLINEAR_NOTEARS_VALID_PARAMS = {
    'model_type': ['mlp', 'sob']
}

# mcsl
MCSL_VALID_PARAMS = {
    'model_type': ['nn', 'qr']
}

# direct lingam
DIRECT_LINGAM_VALID_PARAMS = {
    'measure': ['pwling' , 'kernel']
}

# pc
PC_VALID_PARAMS = {
    'variant': ['original', 'stable', 'parallel'],
    'ci_test': ['fisher', 'g2', 'chi2']
}

# TTPM
TTPM_VALID_PARAMS = {
    'penalty': ['BIC', 'AIC']
}

# DAG_GNN
GNN_VALID_PARAMS = {
    'encoder_type': ['mlp', 'sem'],
    'decoder_type': ['mlp', 'sem'],
    'optimizer': ['adam', 'sgd']
}
