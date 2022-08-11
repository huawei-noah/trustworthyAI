# coding = utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from castle.common import BaseLearner, Tensor
from castle.common import consts
from castle.common.validator import check_args_value

from castle.algorithms.gradient.dag_gnn.torch.utils import functions as func
from castle.algorithms.gradient.dag_gnn.torch.models.modules import Encoder, Decoder


def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


class DAG_GNN(BaseLearner):
    """DAG Structure Learning with Graph Neural Networks

    References
    ----------
    https://arxiv.org/pdf/1904.10098.pdf

    Parameters
    ----------
    encoder_type: str, default: 'mlp'
        choose an encoder, 'mlp' or 'sem'.
    decoder_type: str, detault: 'mlp'
        choose a decoder, 'mlp' or 'sem'.
    encoder_hidden: int, default: 64
        MLP encoder hidden layer dimension, just one hidden layer.
    latent_dim: int, default equal to input dimension
        encoder output dimension
    decoder_hidden: int, default: 64
        MLP decoder hidden layer dimension, just one hidden layer.
    encoder_dropout: float, default: 0.0
        Dropout rate (1 - keep probability).
    decoder_dropout: float, default: 0.0
        Dropout rate (1 - keep probability).
    epochs: int, default: 300
        train epochs
    k_max_iter: int, default: 1e2
        the max iteration number for searching lambda and c.
    batch_size: int, default: 100
        Sample size of each training batch
    lr: float, default: 3e-3
        learning rate
    lr_decay: int, default: 200
        Period of learning rate decay.
    gamma: float, default: 1.0
        Multiplicative factor of learning rate decay.
    lambda_a: float, default: 0.0
        coefficient for DAG constraint h(A).
    c_a: float, default: 1.0
        coefficient for absolute value h(A).
    c_a_thresh: float, default: 1e20
        control loop by c_a
    eta: int, default: 10
        use for update c_a, greater equal than 1.
    multiply_h: float, default: 0.25
        use for judge whether update c_a.
    tau_a: float, default: 0.0
        coefficient for L-1 norm of A.
    h_tolerance: float, default: 1e-8
        the tolerance of error of h(A) to zero.
    use_a_connect_loss: bool, default: False
        flag to use A connect loss
    use_a_positiver_loss: bool, default: False
        flag to enforce A must have positive values
    graph_threshold: float, default: 0.3
        threshold for learned adjacency matrix binarization.
        greater equal to graph_threshold denotes has causal relationship.
    optimizer: str, default: 'Adam'
        choose optimizer, 'Adam' or 'SGD'
    seed: int, default: 42
        random seed
    device_type: str, default: cpu
        ``cpu`` or ``gpu``
    device_ids: int or str, default None
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.

    Examples
    --------
    >>> from castle.algorithms.gradient.dag_gnn.torch import DAG_GNN
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> m = DAG_GNN()
    >>> m.learn(X)
    >>> GraphDAG(m.causal_matrix, true_dag)
    >>> met = MetricsDAG(m.causal_matrix, true_dag)
    >>> print(met.metrics)
    """

    @check_args_value(consts.GNN_VALID_PARAMS)
    def __init__(self, encoder_type='mlp', decoder_type='mlp',
                 encoder_hidden=64, latent_dim=None, decoder_hidden=64,
                 encoder_dropout=0.0, decoder_dropout=0.0, epochs=300, k_max_iter=1e2, tau_a=0.0,
                 batch_size=100, lr=3e-3, lr_decay=200, gamma=1.0, init_lambda_a=0.0, init_c_a=1.0,
                 c_a_thresh=1e20, eta=10, multiply_h=0.25, h_tolerance=1e-8,
                 use_a_connect_loss=False, use_a_positiver_loss=False, graph_threshold=0.3,
                 optimizer='adam', seed=42, device_type='cpu', device_ids='0'):
        super(DAG_GNN, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder_hidden = encoder_hidden
        self.latent_dim = latent_dim
        self.decoder_hidden = decoder_hidden
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.epochs = epochs
        self.k_max_iter = int(k_max_iter)
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.init_lambda_a = init_lambda_a
        self.init_c_a = init_c_a
        self.c_a_thresh = c_a_thresh
        self.eta = eta
        self.multiply_h = multiply_h
        self.tau_a = tau_a
        self.h_tolerance = h_tolerance
        self.use_a_connect_loss = use_a_connect_loss
        self.use_a_positiver_loss = use_a_positiver_loss
        self.graph_threshold = graph_threshold
        self.optimizer = optimizer
        self.seed = seed
        self.device_type = device_type
        self.device_ids = device_ids

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

        self.input_dim = None

    def learn(self, data, columns=None, **kwargs):

        set_seed(self.seed)

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)
        self.n_samples, self.n_nodes, self.input_dim = data.shape

        if self.latent_dim is None:
            self.latent_dim = self.input_dim
        train_loader = func.get_dataloader(data, batch_size=self.batch_size, device=self.device)

        # =====initialize encoder and decoder=====
        adj_A = torch.zeros((self.n_nodes, self.n_nodes), requires_grad=True, device=self.device)
        self.encoder = Encoder(input_dim=self.input_dim,
                               hidden_dim=self.encoder_hidden,
                               output_dim=self.latent_dim,
                               adj_A=adj_A,
                               device=self.device,
                               encoder_type=self.encoder_type.lower()
                               )
        self.decoder = Decoder(input_dim=self.latent_dim,
                               hidden_dim=self.decoder_hidden,
                               output_dim=self.input_dim,
                               device=self.device,
                               decoder_type=self.decoder_type.lower()
                               )
        # =====initialize optimizer=====
        if self.optimizer.lower() == 'adam':
            optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                    {'params': self.decoder.parameters()}],
                                   lr=self.lr)
        elif self.optimizer.lower() == 'sgd':
            optimizer = optim.SGD([{'params': self.encoder.parameters()},
                                   {'params': self.decoder.parameters()}],
                                  lr=self.lr)
        else:
            raise
        self.scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay, gamma=self.gamma)

        ################################
        # main training
        ################################
        c_a = self.init_c_a
        lambda_a = self.init_lambda_a
        h_a_new = torch.tensor(1.0)
        h_a_old = np.inf
        elbo_loss = np.inf
        best_elbo_loss = np.inf
        origin_a = adj_A
        epoch = 0
        for step_k in range(self.k_max_iter):
            while c_a < self.c_a_thresh:
                for epoch in range(self.epochs):
                    elbo_loss, origin_a = self._train(train_loader=train_loader,
                                                      optimizer=optimizer,
                                                      lambda_a=lambda_a,
                                                      c_a=c_a)
                    if elbo_loss < best_elbo_loss:
                        best_elbo_loss = elbo_loss
                if elbo_loss > 2 * best_elbo_loss:
                    break
                # update parameters
                a_new = origin_a.detach().clone()
                h_a_new = func._h_A(a_new, self.n_nodes)
                if h_a_new.item() > self.multiply_h * h_a_old:
                    c_a *= self.eta  # eta
                else:
                    break
            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_a_old = h_a_new.item()
            logging.info(f"Iter: {step_k}, epoch: {epoch}, h_new: {h_a_old}")
            lambda_a += c_a * h_a_new.item()
            if h_a_old <= self.h_tolerance:
                break

        origin_a = origin_a.detach().cpu().numpy()
        origin_a[np.abs(origin_a) < self.graph_threshold] = 0
        origin_a[np.abs(origin_a) >= self.graph_threshold] = 1

        self.causal_matrix = Tensor(origin_a, index=columns, columns=columns)

    def _train(self, train_loader, optimizer, lambda_a, c_a):

        self.encoder.train()
        self.decoder.train()

        # update optimizer
        optimizer, lr = func.update_optimizer(optimizer, self.lr, c_a)

        nll_train = []
        kl_train = []
        origin_a = None
        for batch_idx, (data, relations) in enumerate(train_loader):
            x = Variable(data).double()

            optimizer.zero_grad()

            logits, origin_a = self.encoder(x)
            z_gap = self.encoder.z
            z_positive = self.encoder.z_positive
            wa = self.encoder.wa

            x_pred = self.decoder(logits, adj_A=origin_a, wa=wa)    # X_hat

            # reconstruction accuracy loss
            loss_nll = func.nll_gaussian(x_pred, x)

            # KL loss
            loss_kl = func.kl_gaussian_sem(logits)

            # ELBO loss:
            loss = loss_kl + loss_nll

            # add A loss
            one_adj_a = origin_a  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = self.tau_a * torch.sum(torch.abs(one_adj_a))

            # other loss term
            if self.use_a_connect_loss:
                connect_gap = func.a_connect_loss(one_adj_a, self.graph_threshold, z_gap)
                loss += lambda_a * connect_gap + 0.5 * c_a * connect_gap * connect_gap

            if self.use_a_positiver_loss:
                positive_gap = func.a_positive_loss(one_adj_a, z_positive)
                loss += .1 * (lambda_a * positive_gap
                              + 0.5 * c_a * positive_gap * positive_gap)
            # compute h(A)
            h_A = func._h_A(origin_a, self.n_nodes)
            loss += (lambda_a * h_A
                     + 0.5 * c_a * h_A * h_A
                     + 100. * torch.trace(origin_a * origin_a)
                     + sparse_loss)  # +  0.01 * torch.sum(variance * variance)
            if np.isnan(loss.detach().cpu().numpy()):
                raise ValueError(f"The loss value is Nan, "
                                 f"suggest to set optimizer='adam' to solve it. "
                                 f"If you already set, please check your code whether has other problems.")
            loss.backward()
            optimizer.step()
            self.scheduler.step()

            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())

        return (np.mean(np.mean(kl_train) + np.mean(nll_train)), origin_a)
