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

import random
import numpy as np
import tensorflow as tf


def is_cuda_available():
    return tf.test.is_gpu_available(cuda_only=True)


def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def sample_gumbel(shape, seed, tf_float_type=tf.dtypes.float32, eps=1e-20):
    U = tf.random.uniform(shape, minval=0, maxval=1, dtype=tf_float_type, seed=seed)
    return -tf.math.log(-tf.math.log(U + eps) + eps)


def gumbel_sigmoid(logits, temperature, seed, tf_float_type=tf.dtypes.float32):
    gumbel_softmax_sample = logits \
                            + sample_gumbel(tf.shape(logits), seed, tf_float_type) \
                            - sample_gumbel(tf.shape(logits), seed + 1, tf_float_type)
    y = tf.math.sigmoid(gumbel_softmax_sample / temperature)
    return y


def generate_upper_triangle_indices(d):
    mat = np.arange((d - 1)**2).reshape(d - 1, d - 1)
    target_indices = mat[np.triu(mat, 1) != 0]
    return target_indices


def tensor_description(var):
    """
    Returns a compact and informative string about a tensor.
    Args:
      var: A tensor variable.
    Returns:
      a string with type and size, e.g.: (float32 1x8x8x1024).
    
    Referred from:
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/model_analyzer.py
    """
    description = '(' + str(var.dtype.name) + ' '
    sizes = var.get_shape()
    for i, size in enumerate(sizes):
        description += str(size)
        if i < len(sizes) - 1:
            description += 'x'
    description += ')'
    return description


def print_summary(print_func):
    """
    Print a summary table of the network structure
    Referred from:
    - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/model_analyzer.py
    """
    variables = tf.trainable_variables()

    print_func('Model summary:')
    print_func('---------')
    print_func('Variables: name (type shape) [size]')
    print_func('---------')

    total_size = 0
    total_bytes = 0
    for var in variables:
        # if var.num_elements() is None or [] assume size 0.
        var_size = var.get_shape().num_elements() or 0
        var_bytes = var_size * var.dtype.size
        total_size += var_size
        total_bytes += var_bytes

        print_func('{} {} [{}, bytes: {}]'.format(var.name, tensor_description(var), \
                                                  var_size, var_bytes))

    print_func('Total size of variables: {}'.format(total_size))
    print_func('Total bytes of variables: {}'.format(total_bytes))