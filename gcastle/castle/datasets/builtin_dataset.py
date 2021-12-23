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
import urllib
import tarfile
import hashlib
import pandas as pd
import numpy as np
from urllib.error import URLError

from .simulator import DAG, IIDSimulation
from .simulator import Topology, THPSimulation

USER_AGENT = "gcastle/dataset"


def _check_exist(root, filename, files):
    path_exist = os.path.join(root, filename.split('.')[0])
    processed_folder_exists = os.path.exists(path_exist)
    if not processed_folder_exists:
        return False

    return all(
        _check_integrity(os.path.join(path_exist, file)) for file in files
    )


def _check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True

    md5f = hashlib.md5()
    with open(fpath, 'rb') as f:
        md5f.update(f.read())

    return md5 == md5f.hexdigest()


def _read_data(root, filename, files):
    path_exist = os.path.join(root, filename.split('.')[0])

    result = []
    for file in files:
        if file.split('.')[-1] == 'csv':
            file_path = os.path.join(path_exist, file)
            result.append(pd.read_csv(file_path))
        elif file.split('.')[-1] == 'npy':
            file_path = os.path.join(path_exist, file)
            result.append(np.load(file_path))

    if len(result) == 2:
        result.append(None)

    return result


def _download(root, url, filename, md5):
    """Download the datasets if it doesn't exist already."""

    os.makedirs(root, exist_ok=True)

    # download files
    for mirror in url:
        filepath = "{}{}".format(mirror, filename)
        savegz = os.path.join(root, filename)
        try:
            print("Downloading {}".format(filepath))
            response = urllib.request.urlopen( \
                urllib.request.Request( \
                    filepath, headers={"User-Agent": USER_AGENT}))
            with open(savegz, "wb") as fh:
                fh.write(response.read())

            tar = tarfile.open(savegz)
            names = tar.getnames()
            for name in names:
                tar.extract(name, path=root)
            tar.close()
        except URLError as error:
            print("Failed to download (trying next):\n{}".format(error))
            continue
        break
    else:
        raise RuntimeError("Error downloading {}".format(filename))

    # check integrity of downloaded file
    if not _check_integrity(savegz, md5):
        raise RuntimeError("File not found or corrupted.")


class BuiltinDataSet(object):

    def __init__(self):
        self._data = None
        self._true_graph_matrix = None
        self._topology_matrix = None

    def load(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def data(self):
        return self._data

    @property
    def true_graph_matrix(self):
        return self._true_graph_matrix

    @property
    def topology_matrix(self):
        return self._topology_matrix


class IID_Test(BuiltinDataSet):
    """
    A function for loading IID dataset
    """

    def __init__(self):
        super().__init__()

    def load(self, *args, **kwargs):
        weighted_random_dag = DAG.erdos_renyi(n_nodes=10, n_edges=20,
                                              weight_range=(0.5, 2.0),
                                              seed=1)
        dataset = IIDSimulation(W=weighted_random_dag, n=2000,
                                method='linear', sem_type='gauss')
        self._true_graph_matrix, self._data = dataset.B, dataset.X


class THP_Test(BuiltinDataSet):
    """
    A function for loading THP dataset
    """

    def __init__(self):
        super().__init__()

    def load(self, *args, **kwargs):
        self._true_graph_matrix = DAG.erdos_renyi(n_nodes=10, n_edges=10)
        self._topology_matrix = Topology.erdos_renyi(n_nodes=20, n_edges=20)
        simulator = THPSimulation(self._true_graph_matrix, self._topology_matrix,
                                  mu_range=(0.00005, 0.0001),
                                  alpha_range=(0.005, 0.007))
        self._data = simulator.simulate(T=25000, max_hop=2)


class RealDataSet(BuiltinDataSet):

    def __init__(self):
        super().__init__()
        self.url = None
        self.tar_file = None
        self.md5 = None
        self.file_list = None

    def load(self, root=None, download=False):

        if root is None:
            root = './'

        if _check_exist(root, self.tar_file, self.file_list):
            self._data, self._true_graph_matrix, self._topology_matrix = \
                _read_data(root, self.tar_file, self.file_list)
            return

        if download:
            _download(root, self.url, self.tar_file, self.md5)

        if not _check_exist(root, self.tar_file, self.file_list):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it.')

        self._data, self._true_graph_matrix, self._topology_matrix = \
            _read_data(root, self.tar_file, self.file_list)


class V18_N55_Wireless(RealDataSet):
    """
    A function for loading the real dataset: V18_N55_Wireless
    url: https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/18V_55N_Wireless.tar.gz
    """

    def __init__(self):
        super().__init__()
        self.url = ['https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/']
        self.tar_file = "18V_55N_Wireless.tar.gz"
        self.md5 = "36ee135b86c8dbe09668d9284c23575b"
        self.file_list = ['Alarm.csv', 'DAG.npy']


class V24_N439_Microwave(RealDataSet):
    """
    A function for loading the real dataset: V24_N439_Microwave
    url: https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/24V_439N_Microwave.tar.gz
    """
    
    def __init__(self):
        super().__init__()
        self.url = ['https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/']
        self.tar_file = "24V_439N_Microwave.tar.gz"
        self.md5 = "b4c8b32d34c04a86aa93c7259f7d086c"
        self.file_list = ['Alarm.csv', 'DAG.npy', 'Topology.npy']


class V25_N474_Microwave(RealDataSet):
    """
    A function for loading the real dataset: V25_N474_Microwave
    url: https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/25V_474N_Microwave.tar.gz
    """

    def __init__(self):
        super().__init__()
        self.url = ['https://raw.githubusercontent.com/gcastle-hub/dataset/master/alarm/']
        self.tar_file = "25V_474N_Microwave.tar.gz"
        self.md5 = "51f43ed622d4b44ef6daf8fabf81e162"
        self.file_list = ['Alarm.csv', 'DAG.npy', 'Topology.npy']


class DataSetRegistry(object):
    '''
    A class for resgistering the datasets, in which each dataset
    can be loaded by 'load_dataset' api.
    '''

    meta = {'IID_Test': IID_Test,
            'THP_Test': THP_Test,
            'V18_N55_Wireless': V18_N55_Wireless,
            'V24_N439_Microwave': V24_N439_Microwave,
            'V25_N474_Microwave': V25_N474_Microwave}
