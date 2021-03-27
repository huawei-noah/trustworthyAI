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

import os
import torch
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel

from .base import NonlinearGauss
from .base import NonlinearGaussANM
from .base import compute_constraint
from .base import compute_jacobian_avg
from .base import is_acyclic
from .utils import Accessor
from castle.metrics import MetricsDAG

from castle.common import BaseLearner, Tensor


class Parameters(object):
    """
    This class for saving parameters for the GraN_DAG algorithm.

    Parameters
    ----------
    input_dim : int
        number of input layer, must be int
    learning_folder : str
        name of folder for save learning data
    hidden_num : int
        number of hidden layers
    hidden_dim : int
        number of dimension per hidden layer
    lr : float
        learning rate
    iterations : int
        times of iteration
    optimizer : str, `rmsprop` or `sgd`
        Method of optimize
    h_threshold : float
        constrained threshold
    normalize : bool, default False
        whether normalize data
    gpu : bool, default False
        whether use gpu
    precision : bool, default False
        whether use Double precision
        if True: use torch.FloatTensor
        if False: use torch.DoubleTensor
    batch_size : int
        batch size of per training of NN
    random_seed : int
        random seed
    model : str, default NonlinearGaussANM
        name of model, 'NonlinearGauss' or 'NonlinearGaussANM'
    nonlinear : str, default leaky-relu
        name of Nonlinear activation function
    norm_prod : str, default 'paths'
        use norm of product of paths
    square_prod : bool, default False
        use squared product of paths
    jac_thresh : bool, default True
        get the average Jacobian with the trained model
    lambda_init : float, default 0.0
        initialization of Lagrangian coefficient in the optimization of
        augmented Lagrangian
    mu_init : float, default 0.001
        initialization of penalty coefficient in the optimization of
        augmented Lagrangian
    num_neighbors : None or int, default None
        variable number or neighbors number used in the pns;
        default = variable size
    omega_lambda : float, default 0.0001
        tolerance on the delta lambda, to find saddle points
    omega_mu : float, default 0.9
        check whether the constraint decreases sufficiently if it decreases
        at least (1-omega_mu) * h_prev
    stop_crit_win : int, default 100
        number of iterations for updating values
    edge_clamp_range : float, default 0.0001
        threshold for keeping the edge (if during training
    pns : bool, default False
        whether use pns before training
    pns_thresh : float, default 0.75
        threshold for feature importance score in pns
    train : bool, default True
        whether training model
    to_dag : bool, default True
        whether run to_dag function
    """

    def __init__(self,
                 input_dim,
                 learning_folder='learning_data',
                 hidden_num=2,
                 hidden_dim=10,
                 lr=0.001,
                 iterations=100000,
                 optimizer='rmsprop',
                 h_threshold=1e-8,
                 normalize=False,
                 gpu=False,
                 precision=False,
                 batch_size=64,
                 random_seed=42,
                 model='NonLinGaussANM',
                 nonlinear='leaky-relu',
                 norm_prod='paths',
                 square_prod=False,
                 jac_thresh=True,
                 lambda_init=0.0,
                 mu_init=0.001,
                 num_neighbors=None,
                 omega_lambda=0.0001,
                 omega_mu=0.9,
                 stop_crit_win=100,
                 edge_clamp_range=0.0001,
                 pns=False,
                 pns_thresh=0.75,
                 train=True,
                 to_dag=True):
        self.learning_folder = learning_folder
        self.learning_path = None
        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.iterations = iterations
        self.optimizer = optimizer
        self.h_threshold = h_threshold
        self.normalize = normalize
        self.gpu = gpu
        self.precision = precision
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.model = model
        self.nonlinear = nonlinear
        self.norm_prod = norm_prod
        self.square_prod = square_prod
        self.jac_thresh = jac_thresh
        self.lambda_init = lambda_init
        self.mu_init = mu_init
        self.num_neighbors = num_neighbors
        self.omega_lambda = omega_lambda
        self.omega_mu = omega_mu
        self.stop_crit_win = stop_crit_win
        self.edge_clamp_range = edge_clamp_range
        self.pns = pns
        self.pns_thresh = pns_thresh
        self.train = train
        self.to_dag = to_dag


class GraN_DAG(BaseLearner):
    """
    Gradient Based Neural DAG Learner

    A gradient-based algorithm using neural network modeling for
    non-linear additive noise data

    References: https://arxiv.org/pdf/1906.02226.pdf

    Parameters
    ----------
    params : Parameters
        a class for parameters

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix
    params : class
        Parameters class
    model : model
        neural network model, NonlinearGauss or NonlinearGaussANM

    Examples
    --------
        Load data
    >>> from castle.datasets import load_dataset
    >>> target, data = load_dataset(name='iid_test')

        Initialize parameters
        you can set all parameters in this object like the following example.
    >>> params = Parameters(input_dim=data.shape[1])
    >>> gnd = GraN_DAG(params=params)
    >>> gnd.learn(data=data, target=target)

        You can print GraN_DAG.model.metrics to see it's evaluation metrics.
        Such as TP, FN, FP, TN, SHD, SID e.g.
    >>> print(gnd.model.metrics)

        Also print GraN_DAG.model.adjacency with torch.Tensor type
        or print GranN_DAG.causal_matrix with numpy.ndarray.
    >>> print(gnd.causal_matrix)
    >>> print(gnd.model.adjacency)
    """

    def __init__(self, params):
        super(GraN_DAG, self).__init__()

        self.params = params

    def learn(self, data, target=None, *args, **kwargs):
        """Set up and run the Gran-DAG algorithm

        Parameters
        ----------
        data: numpy.ndarray or Tensor
            include Tensor.data
        target: numpy.ndarray
            train target
        """

        # Control as much randomness as possible
        torch.manual_seed(self.params.random_seed)
        np.random.seed(self.params.random_seed)

        # Use GPU
        if self.params.gpu:
            if self.params.precision:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            if self.params.precision:
                torch.set_default_tensor_type('torch.FloatTensor')
            else:
                torch.set_default_tensor_type('torch.DoubleTensor')

        # create learning_data path
        self.params.learning_path = os.path.join(os.getcwd(),
                                                 self.params.learning_folder)
        if not os.path.exists(self.params.learning_path):
            os.makedirs(self.params.learning_path)

        # create learning model and ground truth model
        if isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, Tensor):
            data = data.data
        else:
            raise TypeError('The type of tensor must be '
                            'Tensor or numpy.ndarray, but got {}'
                            .format(type(data)))

        if data.shape[1] != self.params.input_dim:
            raise ValueError("The number of variables is `{}`, "
                             "the param input_dim is `{}`, "
                             "they must be consistent"
                             ".".format(data.shape[1], self.params.input_dim))
        if self.params.model == "NonLinGauss":
            self.model = NonlinearGauss(input_dim=self.params.input_dim,
                                        hidden_num=self.params.hidden_num,
                                        hidden_dim=self.params.hidden_dim,
                                        output_dim=2,
                                        nonlinear=self.params.nonlinear,
                                        norm_prod=self.params.norm_prod,
                                        square_prod=self.params.square_prod)
        elif self.params.model == "NonLinGaussANM":
            self.model = NonlinearGaussANM(input_dim=self.params.input_dim,
                                           hidden_num=self.params.hidden_num,
                                           hidden_dim=self.params.hidden_dim,
                                           output_dim=1,
                                           nonlinear=self.params.nonlinear,
                                           norm_prod=self.params.norm_prod,
                                           square_prod=self.params.square_prod)
        else:
            raise ValueError(
                "self.params.model has to be in {NonLinGauss, NonLinGaussANM}")

        # create NormalizationData
        if target is None:
            train_data = NormalizationData(data, train_size=1.0, train=True)
            test_data = None
        else:
            train_data = NormalizationData(data, target, train=True)
            test_data = NormalizationData(data, target, train=False,
                                          mean=train_data.mean,
                                          std=train_data.std)

        # apply preliminary neighborhood selection if input_dim > 50
        if self.params.pns:
            if self.params.num_neighbors is None:
                num_neighbors = self.params.input_dim
            else:
                num_neighbors = self.params.num_neighbors

            pns(model=self.model, all_samples=data,
                num_neighbors=num_neighbors,
                thresh=self.params.pns_thresh,
                learning_path=self.params.learning_path)

        # train until constraint is sufficiently close to being satisfied
        if self.params.train:
            if os.path.exists(os.path.join(self.params.learning_path, 'pns')):
                try:
                    self.model = Accessor.load(os.path.join(
                        self.params.learning_path, 'pns'), 'pns_model.pkl')
                except:
                    raise
            train(model=self.model,
                  train_data=train_data,
                  test_data=test_data,
                  params=self.params)
        # remove edges until we have a DAG
        if self.params.to_dag:
            if os.path.exists(os.path.join(self.params.learning_path, 'train')):
                try:
                    self.model = Accessor.load(os.path.join(
                        self.params.learning_path, 'train'), 'train_model.pkl')
                except FileNotFoundError:
                    raise FileNotFoundError("The model is not converged. "
                                            "Adjust the parameters and "
                                            "retrain the model.")
            # to_dag
            to_dag(self.model, train_data, test_data, self.params)

        self._causal_matrix = self.model.adjacency.detach().numpy()


class NormalizationData(object):
    """
    Create Normalization Data object

    Parameters
    ----------
    data : numpy.ndarray
        train x
    target : numpy.ndarray, default None
        train y
    normalize : bool, default False
        whether normalization
    mean : float or None default None
        Mean value of normalization
    std : float or None default None
        Standard Deviation of normalization
    shuffle : bool
        whether shuffle
    train_size : float, default 0.8
        ratio of train data for training
    train : bool, default True
        whether training
    random_seed : int
        for set random seed
    """

    def __init__(self, data, target=None, normalize=False, mean=None, std=None,
                 shuffle=False, train_size=0.8, train=True, random_seed=42):
        self.random = np.random.RandomState(random_seed)
        # load target graph
        if target is None:
            self.adjacency = None
        else:
            self.adjacency = torch.as_tensor(target).type(torch.Tensor)

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
        self.data_set = torch.as_tensor(data).type(torch.Tensor)

        # Normalize data
        self.mean, self.std = mean, std
        if normalize:
            if not mean or not std:
                self.mean = torch.mean(self.data_set, 0, keepdim=True)
                self.std = torch.std(self.data_set, 0, keepdim=True)
            self.data_set = (self.data_set - self.mean) / self.std
        self.n_samples = self.data_set.size(0)

    def sample(self, batch_size):
        """sampling from self.dataset

        Parameters
        ----------
        batch_size : int
            batch size of sample

        Returns
        -------
        samples : torch.Tensor
            sample data after sampling
        torch.ones_like(samples): torch.Tensor
        """
        sample_idxs = self.random.choice(np.arange(int(self.n_samples)),
                                         size=(int(batch_size),),
                                         replace=False)
        samples = self.data_set[torch.as_tensor(sample_idxs).long()]

        return samples, torch.ones_like(samples)


def pns_(model_adj, all_samples, num_neighbors, thresh):
    """Preliminary neighborhood selection

    Parameters
    ----------
    model_adj : numpy.ndarray
        adjacency matrix, all element is 1
    all_samples：numpy.ndarray
        2 dimensional array include all samples
    num_neighbors: integer
        variable number or neighbors number you want
    thresh: float
        apply for sklearn.feature_selection.SelectFromModel

    Returns
    -------
    model_adj : numpy.ndarray
        adjacency matrix, after pns process
    """

    num_nodes = all_samples.shape[1]

    for node in range(num_nodes):
        x_other = np.copy(all_samples)
        x_other[:, node] = 0
        extraTree = ExtraTreesRegressor(n_estimators=500)
        extraTree.fit(x_other, all_samples[:, node])
        selected_reg = SelectFromModel(extraTree,
                                       threshold="{}*mean".format(thresh),
                                       prefit=True,
                                       max_features=num_neighbors)
        mask_selected = selected_reg.get_support(indices=False)
        model_adj[:, node] *= mask_selected

    return model_adj


def pns(model, all_samples, num_neighbors, thresh, learning_path):
    """
    Preliminary neighborhood selection
    After pns, just model.adjacency is changed. if nodes > 50, use it.

    Parameters
    ----------
    model: model object
    all_samples：array-like
        2 dimensional array include all samples
    num_neighbors: integer
        variable number or neighbors number you want
    thresh: float
        apply for sklearn.feature_selection.SelectFromModel
    learning_path: str
        save model after pns
    Returns
    -------
    out: model
    """
    # Prepare path for saving results

    save_path = os.path.join(learning_path, "pns")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if already computed
    if os.path.exists(os.path.join(save_path, "pns_adjacency.npy")):
        return Accessor.load(save_path, "pns_model.pkl")

    model_adj = model.adjacency.detach().cpu().numpy()
    model_adj = pns_(model_adj, all_samples, num_neighbors, thresh)
    with torch.no_grad():
        model.adjacency.copy_(torch.Tensor(model_adj))

    # save model as model.pkl after pns
    Accessor.dump_pkl(model, save_path, 'pns_model')
    Accessor.dump_npy(model_adj, save_path, 'pns_adjacency')

    return model


def train(model, train_data, test_data, params):
    """
    Applying augmented Lagrangian to solve the continuous constrained problem.

    Parameters
    ----------
    model: model object
    train_data: NormalizationData object
    test_data: NormalizationData object
    params: Parameters object
        include all parameters you need in this algorithms

    Returns
    -------
    out: model self
    """

    # Prepare path for saving results
    save_path = os.path.join(params.learning_path, "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if already computed
    if os.path.exists(os.path.join(save_path, "train_DAG.npy")):
        return Accessor.load(save_path, "train_model.pkl")

    # initialize stuff for learning loop
    aug_lagrangians = []
    aug_lagrangian_ma = [0.0] * (params.iterations + 1)
    aug_lagrangians_val = []
    grad_norms = []
    grad_norm_ma = [0.0] * (params.iterations + 1)

    w_adjs = np.zeros((params.iterations,
                       params.input_dim,
                       params.input_dim), dtype=np.float32)

    hs = []
    not_nlls = []  # Augmented Lagrangian minus (pseudo) NLL
    nlls = []  # NLL on train
    nlls_val = []  # NLL on validation

    # Augmented Lagrangian stuff
    mu = params.mu_init
    lamb = params.lambda_init
    mus = []
    lambdas = []

    if params.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    elif params.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params.lr)
    else:
        raise NotImplementedError("optimizer {} is not implemented"
                                  .format(params.optimizer))

    # Learning loop:
    for iter in range(params.iterations):
        # compute loss
        model.train()
        x, _ = train_data.sample(params.batch_size)
        # Initialize weights and bias
        weights, biases, extra_params = model.get_parameters(mode="wbx")
        loss = - torch.mean(model.compute_log_likelihood(x, weights, biases, extra_params))
        nlls.append(loss.item())
        model.eval()

        # constraint related
        w_adj = model.get_w_adj()
        h = compute_constraint(model, w_adj)

        # compute augmented Lagrangian
        aug_lagrangian = loss + 0.5 * mu * h ** 2 + lamb * h

        # optimization step on augmented lagrangian
        optimizer.zero_grad()
        aug_lagrangian.backward()
        optimizer.step()

        # clamp edges
        if params.edge_clamp_range != 0:
            with torch.no_grad():
                to_keep = (w_adj > params.edge_clamp_range) * 1
                model.adjacency *= to_keep

        # logging
        w_adjs[iter, :, :] = w_adj.detach().cpu().numpy().astype(np.float32)
        mus.append(mu)
        lambdas.append(lamb)
        not_nlls.append(0.5 * mu * h.item() ** 2 + lamb * h.item())

        # compute augmented lagrangian moving average
        aug_lagrangians.append(aug_lagrangian.item())
        aug_lagrangian_ma[iter + 1] = aug_lagrangian_ma[iter] + \
                                      0.01 * (aug_lagrangian.item() -
                                              aug_lagrangian_ma[iter])
        grad_norms.append(model.get_grad_norm("wbx").item())
        grad_norm_ma[iter + 1] = grad_norm_ma[iter] + \
                                 0.01 * (grad_norms[-1] - grad_norm_ma[iter])

        # compute loss on whole validation set
        if iter % params.stop_crit_win == 0:
            with torch.no_grad():
                x, _ = test_data.sample(test_data.n_samples)
                loss_val = - torch.mean(model.compute_log_likelihood(x,
                                                                     weights,
                                                                     biases,
                                                                     extra_params))
                nlls_val.append(loss_val.item())
                aug_lagrangians_val.append([iter, loss_val + not_nlls[-1]])

        # compute delta for lambda
        if iter >= 2 * params.stop_crit_win \
                and iter % (2 * params.stop_crit_win) == 0:
            t0 = aug_lagrangians_val[-3][1]
            t_half = aug_lagrangians_val[-2][1]
            t1 = aug_lagrangians_val[-1][1]

            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if not (min(t0, t1) < t_half < max(t0, t1)):
                delta_lambda = -np.inf
            else:
                delta_lambda = (t1 - t0) / params.stop_crit_win
        else:
            delta_lambda = -np.inf  # do not update lambda nor mu

        # Does the augmented lagrangian converged?
        if h > params.h_threshold:
            # if we have found a stationary point of the augmented loss
            if abs(delta_lambda) < params.omega_lambda or delta_lambda > 0:
                lamb += mu * h.item()

                # Did the constraint improve sufficiently?
                hs.append(h.item())
                if len(hs) >= 2:
                    if hs[-1] > hs[-2] * params.omega_mu:
                        mu *= 10

                # little hack to make sure the moving average is going down.
                with torch.no_grad():
                    gap_in_not_nll = 0.5 * mu * h.item() ** 2 + lamb * h.item() - not_nlls[-1]
                    aug_lagrangian_ma[iter + 1] += gap_in_not_nll
                    aug_lagrangians_val[-1][1] += gap_in_not_nll

                if params.optimizer == "rmsprop":
                    optimizer = torch.optim.RMSprop(model.parameters(),
                                                    lr=params.lr)
                else:
                    optimizer = torch.optim.SGD(model.parameters(),
                                                lr=params.lr)
        else:
            # Final clamping of all edges == 0
            with torch.no_grad():
                to_keep = (w_adj > 0).type(torch.Tensor)
                model.adjacency *= to_keep

            # compute nll on train and validation set
            weights, biases, extra_params = model.get_parameters(mode="wbx")
            x, _ = train_data.sample(train_data.n_samples)
            # Since we do not have a DAG yet, this is not really a negative log likelihood.
            nll_train = - torch.mean(model.compute_log_likelihood(x, weights,
                                                                  biases, extra_params))
            x, _ = test_data.sample(test_data.n_samples)
            nll_validation = - torch.mean(model.compute_log_likelihood(x, weights,
                                                                biases, extra_params))
            # Save
            w_adjs = w_adjs[:iter]
            Accessor.dump_pkl(model, save_path, 'train_model')
            Accessor.dump_pkl(params, save_path, 'params')
            if params.input_dim <= 50:
                Accessor.dump_pkl(w_adjs, save_path, 'w_adjs')
            Accessor.dump_pkl(nll_train, save_path, 'pseudo_nll_train')
            Accessor.dump_pkl(nll_validation, save_path, 'pseudo_nll_validation')
            Accessor.dump_pkl(not_nlls, save_path, 'not_nlls')
            Accessor.dump_pkl(aug_lagrangians, save_path, 'aug_lagrangians')
            Accessor.dump_pkl(aug_lagrangian_ma[:iter], save_path, 'aug_lagrangian_ma')
            Accessor.dump_pkl(aug_lagrangians_val, save_path, 'aug_lagrangians_val')
            Accessor.dump_pkl(grad_norms, save_path, 'grad_norms')
            Accessor.dump_pkl(grad_norm_ma[:iter], save_path, 'grad_norm_ma')
            np.save(os.path.join(save_path, "train_DAG.npy"),
                    model.adjacency.detach().cpu().numpy())

            return model


def to_dag(model, train_data, test_data, params, stage_name='to_dag'):
    """
    1- If some entries of A_\phi == 0, also mask them
    (This can happen with stochastic proximal gradient descent)
    2- Remove edges (from weaker to stronger) until a DAG is obtained.

    Parameters
    ----------
    model : model class
        NonlinearGauss or NonlinearGaussANM
    train_data : NormalizationData
        training data
    test_data : NormalizationData
        test data
    params : Parameters
        Parameters class include all parameters
    stage_name : str
        name of folder for saving data after to_dag process

    Returns
    -------
    out : model
    """

    # Prepare path for saving results
    save_path = os.path.join(params.learning_path, stage_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if already computed
    if os.path.exists(os.path.join(save_path, "infer_DAG.npy")):
        return Accessor.load(save_path, "dag_model.pkl")

    model.eval()

    if params.jac_thresh:
        A = compute_jacobian_avg(model, train_data, train_data.n_samples).t()
    else:
        A = model.get_w_adj()
    A = A.detach().cpu().numpy()

    with torch.no_grad():
        # Find the smallest threshold that removes all cycle-inducing edges
        thresholds = np.unique(A)
        epsilon = 1e-8
        for step, t in enumerate(thresholds):
            to_keep = torch.Tensor(A > t + epsilon)
            new_adj = model.adjacency * to_keep
            if is_acyclic(new_adj):
                model.adjacency.copy_(new_adj)
                break

    # evaluate on validation set
    x, _ = test_data.sample(test_data.n_samples)
    weights, biases, extra_params = model.get_parameters(mode="wbx")
    nll_validation = - torch.mean(model.compute_log_likelihood(x, weights,
                                                               biases, extra_params)).item()
    # Compute SHD and SID metrics
    pred_adj_ = model.adjacency.detach().cpu().numpy()
    train_adj_ = train_data.adjacency.detach().cpu().numpy()

    model.metrics = MetricsDAG(pred_adj_, train_adj_).metrics
    del train_adj_, pred_adj_

    # Save
    Accessor.dump_pkl(model, save_path, 'dag_model')
    Accessor.dump_pkl(params, save_path, 'params')
    Accessor.dump_pkl(nll_validation, save_path, "nll_validation", txt=True)
    np.save(os.path.join(save_path, "infer_DAG"),
            model.adjacency.detach().cpu().numpy())

    return model

