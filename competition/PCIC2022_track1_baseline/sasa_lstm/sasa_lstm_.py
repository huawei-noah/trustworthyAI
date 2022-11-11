import os
import random
import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import argparse

from data_utils import data_generator, data_preprocess

import math

from model import ClaLSTM, SASA


class sasa_lstm_model(object):
    def __init__(self, data_path="../dataset/"):
        self.lstm_model = {'dataset2': None, 'dataset3': None}
        self.sasa_model = {'dataset2': None, 'dataset3': None}
        self.lstm_res = {'dataset2': None, 'dataset3': None}
        self.sasa_res = {'dataset2': None, 'dataset3': None}
        self.data_path = data_path

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        pass

    def init_args_lstm(self, dataset):
        parser = argparse.ArgumentParser()
        data_base_path = self.data_path + dataset
        if dataset == "dataset2":
            parser.add_argument("-epochs", type=int, default=50)    # 50
            parser.add_argument("-batch_size", type=int, default=256)
            parser.add_argument("-dim", type=int, default=300, help="只截取最后dim个时刻")
        elif dataset == "dataset3":
            parser.add_argument("-epochs", type=int, default=200)   # 200
            parser.add_argument("-batch_size", type=int, default=128)
            parser.add_argument("-dim", type=int, default=50, help="只截取最后dim个时刻")

        parser.add_argument('-cuda_device', type=int, default=1, help='which gpu to use')
        parser.add_argument("-seed", type=int, default=0)
        parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
        parser.add_argument('-weight_decay', type=float, default=1e-4)
        parser.add_argument('-data_base_path', type=str, default=data_base_path)
        return parser.parse_args()

    def init_args_sasa(self):
        parser = argparse.ArgumentParser(description='train')
        parser.add_argument('-cuda_device', type=str, default=1, help='which gpu to use ')
        parser.add_argument('-random_seed', type=int, default='88', help='random seed')
        parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.0005]')
        parser.add_argument("-seed", type=int, default=10)
        parser.add_argument("-batch_size", type=int, default=64)
        parser.add_argument("-training_steps", type=int, default=9250)
        parser.add_argument("-thres", type=float, default=0.05)
        parser.add_argument("-window_size", type=int, default=25)
        parser.add_argument("-time_interval", type=int, default=5, help="the time interval between two windows")
        args = parser.parse_args()

        return args

    def train_lstm(self):
        datasets = ['dataset2', 'dataset3']
        for dataset in datasets:
            args = self.init_args_lstm(dataset)
            torch.cuda.set_device(args.cuda_device)
            self.set_seed(args.seed)
            print("-------------------- begin reading -------------------------")
            train_X, train_Y, _ = data_preprocess(data_base_path=args.data_base_path,
                                                  dim=args.dim)
            print("-------------------- finish reading -------------------------")
            n_feat = train_X.shape[2]
            print("num_feat: {}".format(n_feat))

            train_data = Data.TensorDataset(train_X, train_Y)
            train_dataloader = Data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

            model = ClaLSTM(feature_size=n_feat)
            model.cuda()

            loss_func = nn.BCELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            print("----------------------- begin training ----------------------------")
            for i in range(args.epochs):
                loss_list = []
                model.train()
                for seq, labels in train_dataloader:
                    seq = seq.cuda()
                    labels = labels.cuda().float().view(-1, 1)

                    optimizer.zero_grad()
                    y_pred = model(seq)
                    loss = loss_func(y_pred, labels)

                    loss.backward()
                    optimizer.step()
                    loss_list.append(loss.item())

                print("epoch: {}, loss:{:.4f}".format(i, sum(loss_list) / len(loss_list)))
            print("----------------------- end training ----------------------------")
            self.lstm_model[dataset] = model

    def train_sasa(self):
        config = self.init_args_sasa()
        self.set_seed(config.random_seed)
        torch.cuda.set_device(config.cuda_device)
        print("------------loading data-------------")
        for dataset in ['dataset2', 'dataset3']:
            data_base_path = self.data_path + dataset
            src_train_path = os.path.join(data_base_path, "cityA/X.npy")
            tgt_train_path = os.path.join(data_base_path, "cityB/train/X.npy")

            segments_length = list(range(config.time_interval, config.window_size + 1, config.time_interval))
            segments_num = len(segments_length)

            src_train_generator = data_generator(data_path=src_train_path, segments_length=segments_length,
                                                 window_size=config.window_size, test=False,
                                                 batch_size=config.batch_size, is_shuffle=True)

            tgt_train_generator = data_generator(data_path=tgt_train_path, segments_length=segments_length,
                                                 window_size=config.window_size, test=False,
                                                 batch_size=config.batch_size, is_shuffle=True)

            print("------------init model-------------")
            model = SASA(max_len=config.window_size, segments_num=segments_num)
            model = model.cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=4e-7)
            global_step = 0

            print("------------start training-------------")
            while global_step < config.training_steps:

                model.train()
                src_train_batch_x, src_train_batch_y, _ = src_train_generator.__next__()
                tgt_train_batch_x, tgt_train_batch_y, _ = tgt_train_generator.__next__()

                if src_train_batch_x.shape[0] != tgt_train_batch_x.shape[0]:
                    continue

                train_batch_x = np.vstack([src_train_batch_x, tgt_train_batch_x])

                y_true = torch.tensor(np.concatenate([src_train_batch_y, tgt_train_batch_y])).float().cuda()
                train_batch_x = torch.tensor(train_batch_x).cuda()

                batch_y_pred, batch_total_loss, batch_label_loss, batch_total_domain_loss_alpha, batch_total_domain_loss_beta = \
                    model.forward(x=train_batch_x,
                                  labeled_sample_size=src_train_batch_y.shape[0] + tgt_train_batch_y.shape[0],
                                  y_true=y_true)
                optimizer.zero_grad()
                batch_total_loss.backward()
                optimizer.step()
                global_step += 1

                print("global_steps", global_step, "total_loss", batch_total_loss.detach().cpu().numpy())

            self.sasa_model[dataset] = model

    def train(self):
        print("sasa model")
        self.train_sasa()
        print("lstm model")
        self.train_lstm()

    def predict(self):
        datasets = ['dataset2', 'dataset3']
        for dataset in datasets:
            # lstm预测
            lstm_args = self.init_args_lstm(dataset)
            torch.cuda.set_device(lstm_args.cuda_device)
            _, _, lstm_test_X = data_preprocess(data_base_path=lstm_args.data_base_path,
                                                dim=lstm_args.dim)
            lstm_test_dataloader = Data.DataLoader(lstm_test_X, batch_size=lstm_args.batch_size, shuffle=False)
            with torch.no_grad():
                model = self.lstm_model[dataset]
                model.eval()

                prob_list = []
                for test_seq in lstm_test_dataloader:
                    test_seq = test_seq.cuda()
                    test_pred = model(test_seq)

                    prob_list.extend(test_pred.detach().cpu().numpy().tolist())

                self.lstm_res[dataset] = pd.DataFrame(prob_list)

            # sasa预测
            config = self.init_args_sasa()
            torch.cuda.set_device(config.cuda_device)
            data_base_path = self.data_path + dataset
            segments_length = list(range(config.time_interval, config.window_size + 1, config.time_interval))
            tgt_test_path = os.path.join(data_base_path, "cityB/test/X.npy")
            tgt_test_generator = data_generator(data_path=tgt_test_path, segments_length=segments_length,
                                                window_size=config.window_size,
                                                batch_size=1024, test=True,
                                                is_shuffle=False)

            tgt_test_set_size = np.load(tgt_test_path).shape[0]

            tgt_test_epoch = int(math.ceil(tgt_test_set_size / float(1024)))
            tgt_test_y_pred_list = list()
            model = self.sasa_model[dataset]
            model.eval()
            for i in range(tgt_test_epoch):
                with torch.no_grad():
                    test_batch_tgt_x, test_batch_tgt_y, test_batch_tgt_l = tgt_test_generator.__next__()

                    test_batch_tgt_x = torch.tensor(test_batch_tgt_x).cuda()
                    test_batch_tgt_y = torch.tensor(test_batch_tgt_y).float().cuda()

                    batch_tgt_y_pred = model.forward(x=test_batch_tgt_x, labeled_sample_size=test_batch_tgt_x.shape[0],
                                                     y_true=test_batch_tgt_y, test_mode=True)
                    tgt_test_y_pred_list.extend(batch_tgt_y_pred.detach().cpu().numpy())

            tgt_test_y_pred_list = np.asarray(tgt_test_y_pred_list)

            y_prob = tgt_test_y_pred_list
            self.sasa_res[dataset] = pd.DataFrame(y_prob)

        lstm_res_concat = pd.concat([self.lstm_res['dataset2'], self.lstm_res['dataset3']]).values.reshape(-1)
        sasa_res_concat = pd.concat([self.sasa_res['dataset2'], self.sasa_res['dataset3']]).values[:, 1]
        ensemble_prob = 0.5 * lstm_res_concat + 0.5 * sasa_res_concat
        pred_res = np.zeros((ensemble_prob.shape[0], 1))
        pred_res[ensemble_prob >= 0.55] = 1
        pred_res = pd.DataFrame(pred_res)

        return pred_res


    pass