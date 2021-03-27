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

from .analyze_utils import convert_logits_to_sigmoid


def callback_after_training(W_logits,temperature,
                            graph_thres):

    d = W_logits.shape[0]
    W_final = convert_logits_to_sigmoid(W_logits / temperature)    # Our final graph
    W_final[W_final <= graph_thres] = 0    # Thresholding
    W_final[W_final > graph_thres] = 1
    W_final[np.arange(d), np.arange(d)] = 0    # Mask diagonal to 0

    return W_final
