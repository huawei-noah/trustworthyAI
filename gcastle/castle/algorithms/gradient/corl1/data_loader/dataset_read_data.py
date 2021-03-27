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
from sklearn.preprocessing import StandardScaler
import time


class DataGenerator(object):

    def __init__(self, X, solution_dag=None, solution_dag_mask=None, normalize_flag=False, transpose_flag=False):

        self.inputdata = X
        self.full_inputdata = X

        self.datasize, self.d = self.inputdata.shape

        if normalize_flag:
            self.inputdata = StandardScaler().fit_transform(self.inputdata)

        if solution_dag is None:
            gtrue = np.zeros([self.d, self.d])
        else:
            gtrue = solution_dag
            if transpose_flag: 
                gtrue = np.transpose(gtrue)
        # (i,j)=1 => node i -> node j
        self.true_graph = np.int32(np.abs(gtrue) > 1e-3) 

        if solution_dag_mask is None:
            gtrue_mask = np.zeros([self.d, self.d])
        else:
            gtrue_mask = solution_dag_mask
            if transpose_flag:
                gtrue_mask = np.transpose(gtrue_mask)
        self.true_graph_mask = np.int32(np.abs(gtrue_mask) > 1e-3)

    def gen_instance_graph(self, max_length, dimension, test_mode=False):
        seq = np.random.randint(self.datasize, size=(dimension))
        input_ = self.inputdata[seq]
        return input_.T

    # Generate random batch for training procedure
    def train_batch(self, batch_size, max_length, dimension):
        input_batch = []

        for _ in range(batch_size):
            input_= self.gen_instance_graph(max_length, dimension)
            input_batch.append(input_)

        return input_batch
