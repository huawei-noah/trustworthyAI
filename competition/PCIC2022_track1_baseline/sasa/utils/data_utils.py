# coding = utf-8

import random
import numpy as np
from sklearn.utils import shuffle


Random_Seed = 88
random.seed(Random_Seed)


def get_one_hot_label(y):
    n_values = np.max(y) + 1
    one_hot_label = np.eye(n_values)[y]
    return one_hot_label


def data_transform(data_path, window_size, segments_length, test_mode):
    """
    transform data to the shape #[ samples_num, x_dim , segments_num , window_size, 1 ]
    """
    data = np.load(data_path)
    if not test_mode:
        label = np.load(data_path.rstrip("X.npy") + "Y.npy").astype('int')
    else:
        label = np.zeros(data.shape[0]).astype('int')

    feature = []
    sample_size = data.shape[0]

    for sample_index in range(sample_size):
        sample = []
        for length in segments_length:
            a = data[sample_index][-length:]
            a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                       mode='constant')
            sample.append(a)

        # (num_segment, window_size, num_feat)
        sample = np.array(sample)

        # (num_feat, num_segment, window_size, newaxis)
        sample = np.transpose(sample, axes=(2, 0, 1))[:, :, :, np.newaxis]
        feature.append(sample)

    feature, label = np.array(feature).astype(np.float32), np.array(label)
    label = get_one_hot_label(label)
    return feature, label


def data_generator(data_path, window_size, segments_length, batch_size, test_mode, is_shuffle=False):

    print('data preparing..')

    feature, label = data_transform(data_path, window_size, segments_length, test_mode)

    if is_shuffle:
        feature, label = shuffle(feature, label)

    batch_count = 0
    while True:
        if batch_size * batch_count >= len(label):
            feature, label = shuffle(feature, label)
            batch_count = 0

        start_index = batch_count * batch_size
        end_index = min(start_index + batch_size, len(label))
        batch_feature = feature[start_index: end_index]

        batch_label = label[start_index: end_index]
        batch_length = np.array(segments_length * (end_index - start_index))
        batch_count += 1

        yield batch_feature, batch_label, batch_length
