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
from scipy.linalg import expm
from tqdm import tqdm
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore import ops, set_seed
from mindspore import Tensor as MsTensor
from mindspore.nn.loss.loss import LossBase
from mindspore import context, dtype as mstype
from mindspore import dataset as ds

from .base import NonlinearGauss
from .base import NonlinearGaussANM
from .base import compute_jacobian_avg
from .base import is_acyclic
from castle.common import BaseLearner, Tensor


class NormalizationData(object):
    """
    Create Normalization Data object

    Parameters
    ----------
    data: numpy.ndarray
        train x
    normalize: bool, default False
        whether normalization
    mean: float or None, default None
        Mean value of normalization
    std: float or None, default None
        Standard Deviation of normalization
    shuffle: bool, default False
        whether shuffle
    train_size: float, default 0.8
        ratio of train data for training
    train: bool, default True
        whether training
    random_seed: int, default 42
        for set random seed
    """

    def __init__(self, data, normalize=False, mean=None, std=None,
                 shuffle=False, train_size=0.8, train=True, random_seed=42):
        self.random = np.random.RandomState(random_seed)

        shuffle_idx = np.arange(data.shape[0])
        if shuffle:
            self.random.shuffle(shuffle_idx)

        if isinstance(train_size, float):
            train_samples = int(data.shape[0] * train_size)
        else:
            raise TypeError("The param train_size must be float < 1")
        if train:
            data = data[shuffle_idx[: train_samples]]
        else:
            data = data[shuffle_idx[train_samples:]]
        # as tensor
        self.data_set = MsTensor(data).astype(dtype=mstype.float32)

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if mean is None or std is None:
                self.mean = msnp.mean(self.data_set, 0, keepdims=True)
                self.std = msnp.std(self.data_set, 0, keepdims=True)
            self.data_set = (self.data_set - self.mean) / self.std
        self.n_samples = self.data_set.shape[0]

    def sample(self, batch_size):
        """
        sampling from self.dataset

        Parameters
        ----------
        batch_size: int
            batch size of sample

        Returns
        -------
        samples: mindspore.Tensor
            sample data after sampling
        ops.ones_like(samples): mindspore.Tensor
        """
        sample_idxs = self.random.choice(np.arange(int(self.n_samples)),
                                         size=(int(batch_size),),
                                         replace=False)
        samples = self.data_set[MsTensor(sample_idxs, mstype.int32)]

        return samples, ops.ones_like(samples)


class GranLoss(LossBase):
    """
    Loss class for network train
    """
    def __init__(self, reduction='mean'):
        super(GranLoss, self).__init__(reduction)

    def construct(self, inputs, *args):
        output = self.get_loss(inputs)
        return output


class GraNDAG(BaseLearner):
    """
    Gradient Based Neural DAG Learner

    A gradient-based algorithm using neural network modeling for
    non-linear additive noise data

    References: https://arxiv.org/pdf/1906.02226.pdf

    Parameters
    ----------
    input_dim: int
        number of input layer (number of varibles), must be int
    hidden_num: int, default 2
        number of hidden layers
    hidden_dim: int, default 10
        number of dimension per hidden layer
    batch_size: int, default 64
        batch size of per training of NN
    lr: float, default 0.001
        learning rate
    iterations: int, default 10000
        times of iteration
    model_name: str, default 'NonLinGaussANM'
        model name, 'NonLinGauss' or 'NonLinGaussANM'
    nonlinear: str, default 'leaky-relu'
        name of Nonlinear activation function, 'sigmoid' or 'leaky-relu'
    optimizer: str, default 'rmsprop'
        Method of optimize, 'rmsprop' or 'sgd'
    h_threshold: float, default 1e-7
        constrained threshold, if constrained value less than equal h_threshold
        means augmented lagrangian has converged, model will stop trainning
    device_type: str, default 'cpu'
        The target device to run, support 'ascend', 'gpu', and 'cpu'
    device_ids: int, default 0
        ID of the target device,
        the value must be in [0, device_num_per_host-1],
        while device_num_per_host should be no more than 4096
    use_pns: bool, default False
        whether use pns before training, if nodes > 50, use it.
    pns_thresh: float, default 0.75
        threshold for feature importance score in pns
    num_neighbors: int, default None
        number of potential parents for each variables
    normalize: bool, default False
        whether normalize data
    random_seed: int, default 42
        random seed
    norm_prod: str, default 'paths'
        use norm of product of paths, 'none' or 'paths'
        'paths': use norm, 'none': with no norm
    square_prod: bool, default False
        use squared product of paths
    jac_thresh: bool, default True
        get the average Jacobian with the trained model
    lambda_init: float, default 0.0
        initialization of Lagrangian coefficient in the optimization of
        augmented Lagrangian
    mu_init: float, default 0.001
        initialization of penalty coefficient in the optimization of
        augmented Lagrangian
    omega_lambda: float, default 0.0001
        tolerance on the delta lambda, to find saddle points
    omega_mu: float, default 0.9
        check whether the constraint decreases sufficiently if it decreases
        at least (1-omega_mu) * h_prev
    stop_crit_win: int, default 100
        number of iterations for updating values
    edge_clamp_range: float, default 0.0001
        threshold for keeping the edge (if during training)

    Examples
    --------
        Load data
    >>> from castle.datasets import load_dataset
    >>> data, true_dag, _ = load_dataset('IID_Test')

    >>> gnd = GraNDAG(input_dim=data.shape[1])
    >>> gnd.learn(data=data)

        Also print GraN_DAG.model.adjacency with mindspore.Tensor type
        or print GranN_DAG.causal_matrix with numpy.ndarray.
    >>> print(gnd.causal_matrix)
    >>> print(gnd.model.adjacency)
    """

    OUT_DIM = 2
    OUT_DIM_ANM = 1

    def __init__(self, input_dim,
                 hidden_num=2,
                 hidden_dim=10,
                 batch_size=64,
                 lr=0.001,
                 iterations=10000,
                 model_name='NonLinGaussANM',
                 nonlinear='leaky-relu',
                 optimizer='rmsprop',
                 h_threshold=1e-7,
                 device_type='cpu',
                 device_ids=0,
                 use_pns=False,
                 pns_thresh=0.75,
                 num_neighbors=None,
                 normalize=False,
                 random_seed=42,
                 jac_thresh=True,
                 lambda_init=0.0,
                 mu_init=0.001,
                 omega_lambda=0.0001,
                 omega_mu=0.9,
                 stop_crit_win=100,
                 edge_clamp_range=0.0001,
                 norm_prod='paths',
                 square_prod=False):
        super(GraNDAG, self).__init__()

        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = lr
        self.iterations = iterations
        self.model_name = model_name
        self.nonlinear = nonlinear
        self.optimizer = optimizer
        self.h_threshold = h_threshold
        self.device_type = device_type
        self.device_ids = device_ids
        self.use_pns = use_pns
        self.pns_thresh = pns_thresh
        self.num_neighbors = num_neighbors
        self.normalize = normalize
        self.random_seed = random_seed
        self.jac_thresh = jac_thresh
        self.lamb = lambda_init
        self.mu = mu_init
        self.omega_lambda = omega_lambda
        self.omega_mu = omega_mu
        self.stop_crit_win = stop_crit_win
        self.edge_clamp_range = edge_clamp_range
        self.norm_prod = norm_prod
        self.square_prod = square_prod

    def learn(self, data, columns=None, **kwargs):
        """Set up and run the Gran-DAG algorithm

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            include Tensor.data
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        """

        # Control as much randomness as possible
        set_seed(self.random_seed)

        # set context for running environment
        context.set_context(mode=context.PYNATIVE_MODE)
        devices_list = ['CPU', 'GPU', 'Ascend']
        if self.device_type.lower() == 'ascend':
            self.device_type = 'Ascend'
        else:
            self.device_type = self.device_type.upper()
        if self.device_type not in devices_list:
            raise ValueError("Only support 'CPU', 'GPU' and 'Ascend'.")
        context.set_context(device_target=self.device_type,
                            device_id=self.device_ids)

        # create learning model and ground truth model
        data = Tensor(data, columns=columns)

        if data.shape[1] != self.input_dim:
            raise ValueError("The number of variables is `{}`, "
                             "the param input_dim is `{}`, "
                             "they must be consistent.".format(data.shape[1],
                                                               self.input_dim))

        if self.model_name == "NonLinGauss":
            self.model = NonlinearGauss(input_dim=self.input_dim,
                                        hidden_num=self.hidden_num,
                                        hidden_dim=self.hidden_dim,
                                        output_dim=self.OUT_DIM,
                                        mu=self.mu,
                                        lamb=self.lamb,
                                        nonlinear=self.nonlinear,
                                        norm_prod=self.norm_prod,
                                        square_prod=self.square_prod)
        elif self.model_name == "NonLinGaussANM":
            self.model = NonlinearGaussANM(input_dim=self.input_dim,
                                           hidden_num=self.hidden_num,
                                           hidden_dim=self.hidden_dim,
                                           output_dim=self.OUT_DIM_ANM,
                                           mu=self.mu,
                                           lamb=self.lamb,
                                           nonlinear=self.nonlinear,
                                           norm_prod=self.norm_prod,
                                           square_prod=self.square_prod)
        else:
            raise ValueError(
                "model has to be in {NonLinGauss, NonLinGaussANM}")

        # create NormalizationData
        train_data = NormalizationData(data, train=True,
                                       normalize=self.normalize)
        test_data = NormalizationData(data, train=False,
                                      normalize=self.normalize,
                                      mean=train_data.mean,
                                      std=train_data.std)

        # apply preliminary neighborhood selection if input_dim > 50
        if self.use_pns:
            if self.num_neighbors is None:
                num_neighbors = self.input_dim
            else:
                num_neighbors = self.num_neighbors

            self.model = neighbors_selection(model=self.model,
                                             all_samples=data,
                                             num_neighbors=num_neighbors,
                                             thresh=self.pns_thresh)

        # update self.model by train
        self._train(train_data=train_data, test_data=test_data)

        # get DAG by run _to_dag
        self._to_dag(train_data)

        self._causal_matrix = Tensor(self.model.adjacency.asnumpy(),
                                     index=data.columns, columns=data.columns)

    def _train(self, train_data, test_data):
        """
        Applying augmented Lagrangian to solve the continuous constrained problem.

        Parameters
        ----------
        train_data: NormalizationData object
            train samples
        test_data: NormalizationData object
            test samples for validation
        """

        # Initialize stuff for learning loop
        aug_lagrangians_val = []

        hs = []
        not_nlls = []  # Augmented Lagrangian minus (pseudo) NLL

        trainable_para_list = self.model.get_trainable_params()
        if self.optimizer == "sgd":
            optimizer = nn.optim.SGD(trainable_para_list, learning_rate=self.lr)
        elif self.optimizer == "rmsprop":
            optimizer = nn.optim.RMSProp(trainable_para_list, learning_rate=self.lr)
        else:
            raise ValueError("optimizer should be in {'sgd', 'rmsprop'}")

        # Package training information
        net_loss = GranLoss()
        net = nn.WithLossCell(self.model, net_loss)
        net = nn.TrainOneStepCell(net, optimizer)

        # Learning loop:
        for iter_num in tqdm(range(self.iterations), desc='Training Iterations'):
            x, _ = train_data.sample(self.batch_size)
            ds_data = self._create_dataset(x.asnumpy(), batch_size=self.batch_size)

            w_adj = self.model.get_w_adj()
            expm_input = expm(w_adj.asnumpy())
            h = np.trace(expm_input) - self.input_dim

            # model train
            self.model.set_train(True)
            net(*list(ds_data)[0])
            self.model.set_train(False)

            # clamp edges, thresholding
            if self.edge_clamp_range != 0:
                to_keep = (w_adj > self.edge_clamp_range) * 1
                self.model.adjacency *= to_keep

            # logging
            not_nlls.append(0.5 * self.model.mu * h ** 2
                            + self.model.lamb * h)

            # compute loss on whole validation set
            if iter_num % self.stop_crit_win == 0:
                x, _ = test_data.sample(test_data.n_samples)
                loss_val = - ops.reduce_mean(
                    self.model.compute_log_likelihood(x))
                aug_lagrangians_val.append(
                    [iter_num, loss_val.asnumpy().item() + not_nlls[-1]])

            # compute delta for lambda
            if iter_num >= 2 * self.stop_crit_win \
                    and iter_num % (2 * self.stop_crit_win) == 0:
                t0 = aug_lagrangians_val[-3][1]
                t_half = aug_lagrangians_val[-2][1]
                t1 = aug_lagrangians_val[-1][1]

                # if the validation loss went up and down,
                # do not update lagrangian and penalty coefficients.
                if not (min(t0, t1) < t_half < max(t0, t1)):
                    delta_lambda = -np.inf
                else:
                    delta_lambda = (t1 - t0) / self.stop_crit_win
            else:
                delta_lambda = -np.inf  # do not update lambda nor mu

            # Does the augmented lagrangian converged?
            # if h value less than equal self.h_threshold value,
            # means augmented lagrangian has converged, stop model training
            if h > self.h_threshold:
                # if we have found a stationary point of the augmented loss
                if abs(delta_lambda) < self.omega_lambda or delta_lambda > 0:
                    self.model.lamb += self.model.mu * h

                    # Did the constraint improve sufficiently?
                    hs.append(h)
                    if len(hs) >= 2:
                        if hs[-1] > hs[-2] * self.omega_mu:
                            self.model.mu *= 10

                    # little hack to make sure the moving average is going down.
                    gap_in_not_nll = 0.5 * self.model.mu * h ** 2 +\
                                     self.model.lamb * h - not_nlls[-1]
                    aug_lagrangians_val[-1][1] += gap_in_not_nll

                    trainable_para_list = self.model.get_trainable_params()
                    if self.optimizer == "rmsprop":
                        optimizer = nn.optim.RMSProp(trainable_para_list,
                                                     learning_rate=self.lr)
                    else:
                        optimizer = nn.optim.SGD(trainable_para_list,
                                                 learning_rate=self.lr)
                    net_loss = GranLoss()
                    net = nn.WithLossCell(self.model, net_loss)
                    net = nn.TrainOneStepCell(net, optimizer)
            else:
                # Final clamping of all edges == 0
                to_keep = (w_adj > 0).astype(mstype.float32)
                self.model.adjacency *= to_keep

                return self.model

    def _to_dag(self, train_data):
        """
        threshold: from weaker to stronger in A
        1- If some entries of A less than equal threshold, mask them to zero
        (This can happen with stochastic proximal gradient descent)
        2- Remove edges by threshold until a DAG is obtained.

        Parameters
        ----------
        train_data: NormalizationData
            train samples
        """

        if self.jac_thresh:
            jacobian_adj = compute_jacobian_avg(
                self.model, train_data, train_data.n_samples).transpose()
        else:
            jacobian_adj = self.model.get_w_adj()

        dia_index = msnp.diag_indices(jacobian_adj.shape[0])
        jacobian_adj[dia_index] = 0

        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(jacobian_adj.asnumpy())
        epsilon = 1e-8
        for step, t in enumerate(thresholds):
            to_keep = (jacobian_adj > t + epsilon).astype(dtype=mstype.float32)
            new_adj = self.model.adjacency * to_keep
            if is_acyclic(new_adj):
                self.model.adjacency = new_adj
                break

        return self.model

    @staticmethod
    def _create_dataset(x_data, batch_size=50):
        """
        Create dataset for network training

        Parameters
        ----------
        x_data: numpy.ndarray
        batch_size: batch size of per training of NN

        Returns
        -------
        out: input_data, shape is (batch_size, num_vars)
        """

        def get_data(x):
            for i in range(x.shape[0]):
                yield x[i, :], np.array([0], dtype=np.float32)

        input_data = ds.GeneratorDataset(list(get_data(x_data)),
                                         column_names=['x', 'y'])
        input_data = input_data.batch(batch_size)

        return input_data


def neighbors_selection(model, all_samples, num_neighbors, thresh):
    """
    Preliminary neighborhood selection
    After pns, just model.adjacency is changed. if nodes > 50, use it.

    Parameters
    ----------
    model: model object
    all_samples: array-like
        2 dimensional array include all samples
    num_neighbors: integer
        variable number or neighbors number you want
    thresh: float
        apply for sklearn.feature_selection.SelectFromModel

    Returns
    -------
    out: model
    """

    model_adj = model.adjacency.asnumpy()
    model_adj = _pns(model_adj, all_samples, num_neighbors, thresh)
    model.adjacency = MsTensor(model_adj).copy()

    return model


def _pns(model_adj, all_samples, num_neighbors, thresh):
    """
    Preliminary neighborhood selection

    Parameters
    ----------
    model_adj: numpy.ndarray
        adjacency matrix, all element is 1
    all_samples: numpy.ndarray
        2 dimensional array include all samples
    num_neighbors: integer
        variable number or neighbors number you want
    thresh: float
        apply for sklearn.feature_selection.SelectFromModel

    Returns
    -------
    model_adj: numpy.ndarray
        adjacency matrix, after pns process
    """

    num_nodes = all_samples.shape[1]

    for node in tqdm(range(num_nodes), desc='Preliminary neighborhood selection'):
        x_other = np.copy(all_samples)
        x_other[:, node] = 0
        extra_tree = ExtraTreesRegressor(n_estimators=500)
        extra_tree.fit(x_other, all_samples[:, node])
        selected_reg = SelectFromModel(extra_tree,
                                       threshold="{}*mean".format(thresh),
                                       prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False)
        model_adj[:, node] *= mask_selected

    return model_adj
