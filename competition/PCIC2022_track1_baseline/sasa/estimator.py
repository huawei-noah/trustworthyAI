# coding = utf-8

import os
import math
import random
import numpy as np
import pandas as pd
import torch
from .utils.data_utils import data_generator
from .base.model import SASA


class SASA_Classiffer(object):
    """
    SASA Classiffer

    References
    ----------
    [1]: https://arxiv.org/abs/2205.03554

    Parameters
    ----------
    batch_size: int, default: 128

    training_steps: int, default: 5000

    h_dim: int, default: 50

    dense_dim: int, default: 50

    l2_weight: float, default: 1e-5

    thresh: float, default: 0.05

    window_size: int, default: 25

    time_interval: int, default: 5
        the time interval between two windows
    lr: float, default: 0.005
        learning rate
    drop_prob: float, default: 0.3
        the probability for dropout
    coeff: float, default: 0.1
        the coefficient of domain loss
    cuda_device: str, default: '0'
        which gpu to use, like '0', '0,1', '0,1,2...n'.
    collect_score: bool, default: True
        whether collect the best score to file
    repetition_idx: int, default: 1
        An index used to mark repeated experiment.
        save_dir like model_save/Boiler/SASA/...._repetition_idx
    source_to_target: str, default: 'S2T'
        the source to the target
    seed: int, default: 10
        random seed

    Methods
    -------
    fit: train model use source domain dataset and target domain dataset
        Args:
            src_train_path: str
                path of source train dataset, must contains X.npy and Y.npy.
            tgt_train_path: str
                path of target train dataset, must contains X.npy and Y.npy.
        Notes::
            save 'pth' to SASA_Classiffer.model_save_path.

    predict: predict label of test dataset
        Args:
            test_data_path: str
                path of test dataset, must contains X.npy.
        Notes::
            save predict label to csv

    Examples
    --------
    >>> # if you want to run this example, store your data like the following code.
    >>> src_train_path = os.path.join("Dataset/phase1_TrainData/cityA/X.npy")
    >>> tgt_train_path = os.path.join("Dataset/phase1_TrainData/cityB/train/X.npy")
    >>> tgt_test_path = os.path.join("Dataset/phase1_TestData/X.npy")
    >>>
    >>> clsf = SASA_Classiffer(input_dim=10, training_steps=100)
    >>> clsf.fit(src_train_path, tgt_train_path)
    >>> clsf.predict(tgt_test_path)
    """

    def __init__(self, input_dim, class_num=2, batch_size=128, training_steps=5000,
                 test_per_step=25, early_stop=30, h_dim=50, dense_dim=50, l2_weight=1e-5,
                 thresh=0.05, window_size=25, time_interval=5, lr=0.005, drop_prob=0.3, coeff=0.1,
                 cuda_device='0', collect_score=True, repetition_idx=1, source_to_target='S2T',
                 seed=10, model_save_base_path='model_save', log_save_base_path='log_save'):
        self.input_dim = input_dim
        self.class_num = class_num
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.test_per_step = test_per_step
        self.early_stop = early_stop
        self.h_dim = h_dim
        self.dense_dim = dense_dim
        self.l2_weight = l2_weight
        self.thresh = thresh
        self.window_size = window_size
        self.time_interval = time_interval
        self.lr = lr
        self.drop_prob = drop_prob
        self.coeff = coeff
        self.cuda_device = cuda_device
        self.collect_score = collect_score
        self.repetition_idx = repetition_idx
        self.source_to_target = source_to_target
        self.seed = seed
        self.model_save_base_path = model_save_base_path
        self.log_save_base_path = log_save_base_path
        self.segments_length = list(
            range(self.time_interval, self.window_size + 1, self.time_interval)
        )
        self.segments_num = len(self.segments_length)
        self.model_name = 'SASA'
        self.dataset_name = 'pcic'

        self._build_save_path()
        self._init_random_state()
        self._init_model()

    def _init_random_state(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = self.cuda_device
            self.device = torch.device('cuda')
            torch.cuda.manual_seed(self.seed)
        else:
            self.device = torch.device('cpu')
            torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _init_model(self):

        print("------------init model-------------")
        self.model = SASA(max_len=self.window_size, coeff=self.coeff,
                          segments_num=self.segments_num, input_dim=self.input_dim,
                          class_num=self.class_num, h_dim=self.h_dim, dense_dim=self.dense_dim,
                          drop_prob=self.drop_prob, lstm_layer=1, device=self.device)
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)

    def fit(self, src_train_path, tgt_train_path):
        src_train_generator = data_generator(data_path=src_train_path,
                                             segments_length=self.segments_length,
                                             window_size=self.window_size, test_mode=False,
                                             batch_size=self.batch_size * 2, is_shuffle=True)
        tgt_train_generator = data_generator(data_path=tgt_train_path,
                                             segments_length=self.segments_length,
                                             window_size=self.window_size, test_mode=False,
                                             batch_size=self.batch_size * 2, is_shuffle=True)
        # train model
        print("------------start training-------------")
        global_step = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=4e-7)

        while global_step < self.training_steps:
            self.model.train()
            src_train_batch_x, src_train_batch_y, _ = src_train_generator.__next__()
            tgt_train_batch_x, tgt_train_batch_y, _ = tgt_train_generator.__next__()

            if src_train_batch_x.shape[0] != tgt_train_batch_x.shape[0]:
                continue

            train_batch_x = np.vstack([src_train_batch_x, tgt_train_batch_x])

            y_true = torch.tensor(
                np.concatenate([src_train_batch_y, tgt_train_batch_y])).float().to(self.device)
            train_batch_x = torch.tensor(train_batch_x).to(self.device)

            model_result = self.model.forward(x=train_batch_x,
                                         labeled_sample_size=src_train_batch_y.shape[0] +
                                                             tgt_train_batch_y.shape[0],
                                         y_true=y_true)
            (batch_y_pred, batch_total_loss, batch_label_loss, batch_total_domain_loss_alpha,
             batch_total_domain_loss_beta) = model_result
            optimizer.zero_grad()
            batch_total_loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 50 == 0:
                print("global_steps", global_step, "total_loss",
                      batch_total_loss.detach().cpu().numpy())
        pth_name = os.path.join(self.model_save_path, 'sasa_baseline.pth')
        torch.save(self, pth_name)
        print(f"------------The model be saved in dir: {pth_name}-------------")

    def predict(self, test_data_path):
        """predict data in test_data_path"""

        tgt_test_generator_2 = data_generator(data_path=test_data_path,
                                              segments_length=self.segments_length,
                                              window_size=self.window_size, test_mode=True,
                                              batch_size=1024, is_shuffle=False)
        tgt_test_set_size = 43200
        tgt_test_epoch = int(math.ceil(tgt_test_set_size / float(1024)))
        tgt_test_y_pred_list = list()

        for i in range(tgt_test_epoch):
            self.model.eval()
            with torch.no_grad():
                test_batch_tgt_x, test_batch_tgt_y, test_batch_tgt_l = tgt_test_generator_2.__next__()

                test_batch_tgt_x = torch.tensor(test_batch_tgt_x).to(self.device)
                test_batch_tgt_y = torch.tensor(test_batch_tgt_y).float().to(self.device)

                batch_tgt_y_pred = self.model.forward(x=test_batch_tgt_x,
                                                 labeled_sample_size=test_batch_tgt_y.shape[0],
                                                 y_true=test_batch_tgt_y, test_mode=True)
                tgt_test_y_pred_list.extend(batch_tgt_y_pred.detach().cpu().numpy())
        tgt_test_y_pred_list = np.asarray(tgt_test_y_pred_list)
        y_pred = pd.DataFrame((tgt_test_y_pred_list[:, 1] > 0.5) * 1)

        # save to predict result to csv
        csv_path = "submission_window{}_interval{}.csv".format(self.window_size, self.time_interval)
        y_pred.to_csv(csv_path, header=False, index=False)
        print(f"------------The predict label be saved in dir: {csv_path}-------------")

    def _build_save_path(self):
        # build save path
        self.model_save_path = os.path.join(
            self.model_save_base_path, self.dataset_name, self.model_name,
            f'{self.source_to_target}_hdim{self.h_dim}_ddim{self.dense_dim}_'
            f'wz{self.window_size}_bsz{self.batch_size}_lr{self.lr}_'
            f'dp{self.drop_prob}_coeff{self.coeff}_rept{self.repetition_idx}'
        )
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path, exist_ok=True)
