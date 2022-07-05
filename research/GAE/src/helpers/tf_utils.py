import os
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
    tf.set_random_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


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

        print_func('{} {} [{}, bytes: {}]'.format(var.name, tensor_description(var), var_size, var_bytes))

    print_func('Total size of variables: {}'.format(total_size))
    print_func('Total bytes of variables: {}'.format(total_bytes))
