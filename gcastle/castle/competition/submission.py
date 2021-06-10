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
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def transform(source, target):
    """
    This function is used to generate the submission file for the causality competition:
    https://competition.huaweicloud.com/information/1000041487/circumstance
    
    Parameters
    ----------
    source: str
        The source directory used to store all the .npy files.
    target: str
        The target directory/file used to store the submission file.
    """

    if not os.path.splitext(target)[1] == '.csv':
        target = os.path.join(target, 'submit.csv')

    f_list = os.listdir(source)
    npy_list = []
    for i in f_list:
        if os.path.splitext(i)[1] == '.npy':
            try:
                npy_list.append(int(i.split('.')[0]))
            except:
                pass
    
    if len(npy_list) > 24:
        raise RuntimeError('Number of .npy files is ' + str(len(npy_list)) + '! (must <= 24)')
    elif len(npy_list) == 0:
        raise RuntimeError('Cannot find any .npy file!')

    npy_list.sort()

    for i in range(1, len(npy_list)):
        if npy_list[i] - npy_list[i-1] == 1:
            pass
        else:
            raise RuntimeError('The .npy filenames are not continuous!')

    arrs = []
    for i in npy_list:
        arrs.append(np.load(os.path.join(source, str(i)+'.npy')))

    def arr_to_string(mat):
        mat_int = mat.astype(int)
        mat_flatten = mat_int.flatten().tolist()
        for m in mat_flatten:
            if m not in [0, 1]:
                raise TypeError('Any element in a numpy array is supposed to be 0 or 1.')
        mat_str = ' '.join(map(str, mat_flatten))
        return mat_str

    arrs_str = [arr_to_string(arr) for arr in arrs]
    pd.DataFrame(arrs_str).to_csv(target, index=False)

    logging.info('Successfully generated the submission file: ' + target + '.')
