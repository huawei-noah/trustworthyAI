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

import os
import sys
from pathlib import Path

PROJECT_DIR = str(Path(__file__).resolve().parents[2])
FILE_PATH = os.path.join(PROJECT_DIR, "data")
if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH)

INLINE_DATASETS = list()
INLINE_TRUE = list()

SEM_TYPE = {"IID_LINEAR": ["gauss", "exp", "gumbel", "uniform", "logistic",
                           "poisson"],
            "IID_NONLINEAR": ["mlp", "mim", "gp", "gp-add", "quadratic"],
            "EVENT": []}

lang_str = {'en': 'en_US', 'zh': 'zh_CN'}
