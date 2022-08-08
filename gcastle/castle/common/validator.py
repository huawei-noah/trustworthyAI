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

"""
Module contains many utils for validating data or function arguments
"""

import inspect
import functools
import numpy as np
import torch


def transfer_to_device(*args, device=None):
    """
    Transfer `*args` to `device`

    Parameters
    ----------
    args: np.ndarray, torch.Tensor
        variables that need to transfer to `device`
    device: str
        'cpu' or 'gpu', if None, default='cpu

    Returns
    -------
    out: args
    """

    out = []
    for each in args:
        if isinstance(each, np.ndarray):
            each = torch.tensor(each, device=device)
        elif isinstance(each, torch.Tensor):
            each = each.to(device=device)
        else:
            raise TypeError(f"Expected type of the args is np.ndarray "
                            f"or torch.Tensor, but got `{type(each)}`.")
        out.append(each)
    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]


def check_args_value(compat):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            params = inspect.getfullargspec(func)
            pos_args = args
            if params.args[0] in ['self', 'cls']:
                pos_args = pos_args[1:]
            for i, v in enumerate(pos_args):
                valid = compat[params.args[i]]
                if v not in valid:
                    raise ValueError(f"Invalid value at position [{i}], "
                                     f"expected one of {valid}, but got '{v}'.")
            for key, value in kwargs.items():
                if key not in compat.keys():
                    continue
                valid = compat[key]
                if value not in valid:
                    raise ValueError(f"Invalid value at `{key}`, expected one "
                                     f"of {valid}, but got '{value}'.")
            return func(*args, **kwargs)
        return wrapper
    return decorator
