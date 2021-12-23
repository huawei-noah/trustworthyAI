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

from .builtin_dataset import *


def load_dataset(name='IID_Test', root=None, download=False):
    """
    A function for loading some well-known datasets.

    Parameters
    ----------
    name: class, default='IID_Test'
        Dataset name, independent and identically distributed (IID),
        Topological Hawkes Process (THP) and real datasets.
    root: str
        Root directory in which the dataset will be saved.
    download: bool
        If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.

    Return
    ------
    out: tuple
        true_graph_matrix: numpy.matrix
            adjacency matrix for the target causal graph.
        topology_matrix: numpy.matrix
            adjacency matrix for the topology.
        data: pandas.core.frame.DataFrame
            standard trainning dataset.
    """

    if name not in DataSetRegistry.meta.keys():
        raise ValueError('The dataset {} has not been registered, you can use'
                         ' ''castle.datasets.__builtin_dataset__'' to get registered '
                         'dataset list'.format(name))
    loader = DataSetRegistry.meta.get(name)()
    loader.load(root, download)
    return loader.data, loader.true_graph_matrix, loader.topology_matrix
