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

import math
import numpy as np
from scipy.stats import norm

class CI_Test(object):
    """Class of conditional independence test"""

    @staticmethod
    def gauss_test(data, x, y, ctrl_var):
        """Gauss test

        Parameters
        ----------
        data : array, (n_samples, n_features)
            Dataset
        x : int
            The first node
        y : int
            The second node
        ctrl_var: List
            The set of neighboring nodes of x and y (controlled variables)

        Returns
        -------
        p: float
            the p-value of conditional independence.
        """

        n = data.shape[0]
        k = len(ctrl_var)
        if k == 0:
            r = np.corrcoef(data[:, [x, y]].T)[0][1]
        else:
            sub_index = [x, y]
            sub_index.extend(ctrl_var)
            sub_corr = np.corrcoef(data[:, sub_index].T)
            # inverse matrix
            PM = np.linalg.inv(sub_corr)
            r = -1 * PM[0, 1] / math.sqrt(abs(PM[0, 0] * PM[1, 1]))
        cut_at = 0.99999
        r = min(cut_at, max(-1 * cut_at, r))  # make r between -1 and 1

        # Fisher’s z-transform
        res = math.sqrt(n - k - 3) * .5 * math.log1p((2 * r) / (1 - r))
        p = 2 * (1 - norm.cdf(abs(res)))

        return p

