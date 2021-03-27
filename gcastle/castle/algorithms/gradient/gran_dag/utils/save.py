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
import pickle
import uuid
import numpy as np
import pandas as pd

class Accessor(object):
    """
    accessor
    """

    @staticmethod
    def dump_pkl(obj, path, name, txt=False):
        if not txt:
            with open(os.path.join(path, name + ".pkl"), "wb") as f:
                pickle.dump(obj, f)
        else:
            with open(os.path.join(path, name + ".txt"), "w") as f:
                f.write(str(obj))

    @staticmethod
    def dump_npy(array, path, name):

        save_path = os.path.join(path, name)
        np.save(save_path, array)
        del save_path

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, name), "rb") as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def np_to_csv(array, save_path):
        """
        Convert np array to .csv

        array: numpy array
            the numpy array to convert to csv
        save_path: str
            where to temporarily save the csv
        Return the path to the csv file
        """
        id = str(uuid.uuid4())
        output = os.path.join(os.path.dirname(save_path), 'tmp_' + id + '.csv')

        df = pd.DataFrame(array)
        df.to_csv(output, header=False, index=False)

        return output