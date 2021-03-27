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
import pandas as pd
from tqdm import tqdm


class BaseLearner(object):

    def __init__(self):

        self._causal_matrix = None

    def learn(self, data, *args, **kwargs):

        raise NotImplementedError

    @property
    def causal_matrix(self):
        return self._causal_matrix

    @causal_matrix.setter
    def causal_matrix(self, value):
        self._causal_matrix = value


class Tensor(object):
    """
    A Tensor is a common data structure in gCastle, it's mainly used to
    standardize what we input into the learning algorithms. It supports
    various types data reading, including 'numpy.ndarray', 
    'pandas.DataFrame', etc.

    Parameters
    ----------
    indata: numpy.ndarray or pandas.DataFrame
        Input object, can be a two-dimensional data for ndarray/DataFrame.
    """

    def __init__(self, indata):

        self._set_tensor(indata)

    def _set_tensor(self, indata):
        """
        Set tensor for input data.

        Parameters
        ----------
        indata: numpy.ndarray or pandas.DataFrame
            Input object, can be a 2 or 3 dimensional ndarray or a DataFrame.

        Return
        ------
        tensor: castle.tensor
            Output castle.tensor(data) format object.
        """

        if isinstance(indata, np.ndarray):
            if indata.ndim == 2 or indata.ndim == 3:
                self.data = indata
            else:
                raise TypeError("Input numpy.ndarray ndim error!") 
        elif isinstance(indata, pd.DataFrame):
            self.data = indata.values
        else:
            raise TypeError("Input data is not numpy.ndarray or pd.DataFrame!") 
