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
