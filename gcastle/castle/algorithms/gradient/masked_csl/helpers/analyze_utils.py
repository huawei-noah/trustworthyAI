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

import numpy as np
from scipy.linalg import expm


def convert_logits_to_sigmoid(W, tau=1.0):
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    d = W.shape[0]
    W = np.copy(W)
    sigmoid_W = sigmoid(W/tau)
    sigmoid_W[np.arange(d), np.arange(d)] = 0    # Mask diagonal to 0
    return sigmoid_W


def compute_acyclicity(W):
    return np.trace(expm(W * W)) - W.shape[0]
