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

import abc
import numpy as np
import pandas as pd
from collections.abc import Iterable
from pandas import Index, RangeIndex


class BaseLearner(metaclass=abc.ABCMeta):

    def __init__(self):

        self._causal_matrix = None

    @abc.abstractmethod
    def learn(self, data, *args, **kwargs):

        raise NotImplementedError

    @property
    def causal_matrix(self):
        return self._causal_matrix

    @causal_matrix.setter
    def causal_matrix(self, value):
        self._causal_matrix = value


class Tensor(np.ndarray):
    """A subclass of numpy.ndarray.

    This subclass has all attributes and methods of numpy.ndarray
    with two additional, user-defined attributes: `index` and `columns`.

    It can be used in the same way as a standard numpy.ndarray.
    However, after performing any operations on the Tensor (e.g., slicing,
    transposing, arithmetic, etc.), the user-defined attribute values of
    `index` and `columns` will be lost and replaced with a numeric indices.

    Parameters
    ----------
    object: array-like
        Multiple list, ndarray, DataFrame
    index : Index or array-like
        Index to use for resulting tensor. Will default to RangeIndex if
        no indexing information part of input data and no index provided.
    columns : Index or array-like
        Column labels to use for resulting tensor. Will default to
        RangeIndex (0, 1, 2, ..., n) if no column labels are provided.

    Examples
    --------
    Create a Tensor from a list or numpy.ndarray.

    >>> x = [[0, 3, 8, 1],
    ...      [8, 4, 1, 9],
    ...      [7, 3, 3, 7]]

    Or

    >>> x = np.random.randint(0, 10, size=12).reshape((3, 4))
    >>> arr = Tensor(x)
    >>> arr
    Tensor([[0, 3, 8, 1],
            [8, 4, 1, 9],
            [7, 3, 3, 7]])
    >>> arr.index
    RangeIndex(start=0, stop=3, step=1)
    >>> list(arr.index)
    [0, 1, 2]
    >>> arr.columns
    RangeIndex(start=0, stop=4, step=1)
    >>> list(arr.columns)
    [0, 1, 2, 3]

    `index` and `columns` can be set using kwargs.

    >>> arr = Tensor(x, index=list('XYZ'), columns=list('ABCD'))
    >>> arr
    Tensor([[6, 1, 8, 9],
            [1, 5, 2, 1],
            [5, 9, 4, 5]])
    >>> arr.index
    Index(['x', 'y', 'z'], dtype='object')
    >>> arr.columns
    Index(['a', 'b', 'c', 'd'], dtype='object')

    Or a value can be assigned to `arr.index` or `arr.columns`,
    but it must be an `Iterable`.

    >>> arr.index = list('xyz')
    >>> arr.index
    Index(['x', 'y', 'z'], dtype='object')
    >>> arr.columns = list('abcd')
    >>> arr.columns
    Index(['a', 'b', 'c', 'd'], dtype='object')

    A Tensor can also be created from a pandas.DataFrame.

    >>> x = pd.DataFrame(np.random.randint(0, 10, size=12).reshape((3, 4)),
    ...                  index=list('xyz'),
    ...                  columns=list('abcd'))
    >>> x
       a  b  c  d
    x  6  1  8  9
    y  1  5  2  1
    z  5  9  4  5
    >>> arr = Tensor(x)
    >>> arr
    Tensor([[6, 1, 8, 9],
            [1, 5, 2, 1],
            [5, 9, 4, 5]])
    >>> arr.index
    Index(['x', 'y', 'z'], dtype='object')
    >>> arr.columns
    Index(['a', 'b', 'c', 'd'], dtype='object')

    It's possible to use any method of numpy.ndarray on the Tensor,
    such as `sum`, `@`, etc.

    >>> arr.sum(axis=0)
    Tensor([15, 10, 12, 17])
    >>> arr @ arr.T
    Tensor([[ 74,  29,  40],
            [ 29, 162, 134],
            [ 40, 134, 116]])

    If the Tensor is sliced, the values of `index` and `columns` will disappear,
    and new values of type `RangeIndex` will be created.

    >>> new_arr = arr[:, 1:3]
    >>> new_arr
    Tensor([[1, 8],
            [5, 2],
            [9, 4]])
    >>> new_arr.index
    RangeIndex(start=0, stop=3, step=1)
    >>> new_arr.columns
    RangeIndex(start=0, stop=2, step=1)

    If you want to retain the values of `index` and `columns`,
    you can reassign them.

    >>> new_arr.index = arr.index[:]
    >>> new_arr.index
    Index(['x', 'y', 'z'], dtype='object')

    >>> new_arr.columns = arr.columns[1:3]
    >>> new_arr.columns
    Index(['b', 'c'], dtype='object')

    We recommend performing slicing operations in the following way
    to keep the `index` and `columns` values.

    >>> new_arr = Tensor(array=arr[:, 1:3],
    ...                  index=arr.index[:, 1:3],
    ...                  columns=arr.columns[:, 1:3])
    >>> new_arr.index
    Index(['x', 'y', 'z'], dtype='object')
    >>> new_arr.columns
    Index(['b', 'c'], dtype='object')
    """

    def __new__(cls, object=None, index=None, columns=None):

        if object is None:
            raise TypeError("Tensor() missing required argument 'object' (pos 0)")
        elif isinstance(object, list):
            object = np.array(object)
        elif isinstance(object, pd.DataFrame):
            index = object.index
            columns = object.columns
            object = object.values
        elif isinstance(object, (np.ndarray, cls)):
            pass
        else:
            raise TypeError(
                "Type of the required argument 'object' must be array-like."
            )
        if index is None:
            index = range(object.shape[0])
        if columns is None:
            columns = range(object.shape[1])
        obj = np.asarray(object).view(cls)
        obj.index = index
        obj.columns = columns

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if self.ndim == 0: return
        elif self.ndim == 1:
            self.columns = RangeIndex(0, 1, step=1, dtype=int)
        else:
            self.columns = RangeIndex(0, self.shape[1], step=1, dtype=int)
        self.index = RangeIndex(0, self.shape[0], step=1, dtype=int)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        assert isinstance(value, Iterable)
        if len(list(value)) != self.shape[0]:
            raise ValueError("Size of value is not equal to the shape[0].")
        self._index = Index(value)

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        assert isinstance(value, Iterable)
        if (self.ndim > 1 and len(list(value)) != self.shape[1]):
            raise ValueError("Size of value is not equal to the shape[1].")
        self._columns = Index(value)
