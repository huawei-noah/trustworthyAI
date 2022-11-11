import os

import numpy as np
import torch


from sklearn.utils import shuffle


def normalization(data, test_data=None):
    for feature_index in range(data.shape[2]):
        mean_values = np.mean(data[:, :, feature_index])
        var_values = np.var(data[:, :, feature_index])
        data[:, :, feature_index] = (data[:, :, feature_index] - mean_values) / np.sqrt(var_values)
        if test_data is not None:
            test_data[:, :, feature_index] = (test_data[:, :, feature_index] - mean_values) / np.sqrt(var_values)
    return data, test_data


def data_preprocess(data_base_path,
                    dim, normal=1):
    train_source_X = np.load(data_base_path + "/cityA/X.npy")
    train_source_Y = np.load(data_base_path + "/cityA/Y.npy")
    if normal == 1:
        train_source_X, _ = normalization(train_source_X)
    train_source_X = train_source_X[:, -dim:, :]
    print("finish reading train_source_data, the ratio of neg samples is {:.4f}".format(
        train_source_Y.sum() / train_source_Y.shape[0]))

    train_target_X = np.load(data_base_path + "/cityB/train/X.npy")
    train_target_Y = np.load(data_base_path + "/cityB/train/Y.npy")
    if normal == 1:
        train_target_X, _ = normalization(train_target_X)
    train_target_X = train_target_X[:, -dim:, :]
    print("finish reading train_target_data, the ratio of neg samples is {:.4f}".format(
        train_target_Y.sum() / train_target_Y.shape[0]))

    test_X = np.load(data_base_path + "/cityB/test/X.npy")
    if normal == 1:
        test_X, _ = normalization(test_X)
    test_X = test_X[:, -dim:, :]
    print("finish reading test_data")

    train_X = torch.tensor(np.concatenate([train_source_X, train_target_X])).float()
    train_Y = torch.tensor(np.concatenate([train_source_Y, train_target_Y])).float()
    test_X = torch.tensor(test_X).float()
    print(train_source_X.shape, train_target_X.shape, test_X.shape)

    return train_X, train_Y, test_X


def get_one_hot_label(y):
    n_values = np.max(y) + 1
    one_hot_label = np.eye(n_values)[y]
    return one_hot_label


def data_transform(data_path, window_size, segments_length, test):
    data = np.load(data_path)
    if not test:
        label = np.load(data_path.rstrip("X.npy") + "Y.npy")
    else:
        label =  np.zeros(len(data))

    feature = []
    sample_size = data.shape[0]

    for sample_index in range(sample_size):
        sample = []
        for length in segments_length:
            a = data[sample_index][-length:]
            a = np.pad(a, pad_width=((0, window_size - length), (0, 0)),
                       mode='constant')
            sample.append(a)

        sample = np.array(sample)

        sample = np.transpose(sample, axes=(2, 0, 1))[:, :, :,
                 np.newaxis]

        feature.append(sample)
    feature, label = np.array(feature).astype(np.float32), np.array(label).astype(np.float32)
    label = get_one_hot_label(label.astype(int))
    print(data_path, feature.shape)
    return feature, label


def data_generator(data_path, window_size, segments_length, batch_size, test, is_shuffle=False):
    print('data preparing..')

    feature, label = data_transform(data_path, window_size, segments_length, test=test)

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


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
