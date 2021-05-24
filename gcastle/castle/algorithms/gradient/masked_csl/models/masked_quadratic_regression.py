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
import tensorflow as tf

from .masked_model import MaskedModel
from ..helpers.tf_utils import generate_upper_triangle_indices


class MaskedQuadraticRegression(MaskedModel):
    """
    References:
    - https://realpython.com/python-super/
    - https://www.digitalocean.com/community/tutorials/understanding-class-inheritance-in-python-3
    - https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/2_BasicModels/linear_regression.py
    """

    def _forward(self, x):
        # x is of shape (n, d - 1)
        # Return a vector of shape (n,)
        output = 0

        # Linear terms
        w = tf.Variable(np.random.uniform(low=-0.05, high=0.05, size=(self.d - 1,)), dtype=self.tf_float_type)
        output += tf.reduce_sum(w * x, axis=1)

        # Squared terms
        w = tf.Variable(np.random.uniform(low=-0.05, high=0.05, size=(self.d - 1,)), dtype=self.tf_float_type)
        output += tf.reduce_sum(w * tf.math.square(x), axis=1)

        # Cross terms
        x_ = tf.expand_dims(x, 1)
        y_ = tf.expand_dims(x, 2)
        target_indices = generate_upper_triangle_indices(self.d)
        all_cross_terms = tf.reshape(x_ * y_, (self.n, -1))
        combinations_cross_terms = tf.gather(all_cross_terms, indices=target_indices, axis=1)
        w = tf.Variable(np.random.uniform(low=-0.05, high=0.05, size=(len(target_indices),)), dtype=self.tf_float_type)
        output += tf.reduce_sum(w * combinations_cross_terms, axis=1)

        # Bias term
        # b = tf.Variable(np.random.randn(), dtype=self.tf_float_type)
        # output += b
        return output
