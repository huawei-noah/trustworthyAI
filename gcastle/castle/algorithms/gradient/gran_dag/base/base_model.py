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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions

from .dag_optimizer import compute_A_phi


class BaseModel(nn.Module):
    """Base class of LearnableModel, disable instantiation."""

    def __init__(self, input_dim, hidden_num, hidden_dim, output_dim,
                 nonlinear="leaky-relu", norm_prod='path', square_prod=False):

        super(BaseModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_num = hidden_num
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinear = nonlinear
        self.norm_prod = norm_prod
        self.square_prod = square_prod

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.extra_params = []
        # Those parameter might be learnable, but they do not depend on parents.

        # initialize current adjacency matrix
        self.adjacency = torch.ones((self.input_dim, self.input_dim)) - \
                         torch.eye(self.input_dim)

        self.zero_weights_ratio = 0.
        self.numel_weights = 0

        # Generate layer_list
        layer_list = [self.hidden_dim] * self.hidden_num
        layer_list.insert(0, self.input_dim)
        layer_list.append(self.output_dim)

        # Instantiate the parameters of each layer in the model of each variable
        for i, item in enumerate(layer_list[:-1]):
            in_dim = item
            out_dim = layer_list[i + 1]
            self.weights.append(nn.Parameter(torch.zeros(self.input_dim,
                                                         out_dim,
                                                         in_dim),
                                             requires_grad=True))
            self.biases.append(nn.Parameter(torch.zeros(self.input_dim,
                                                        out_dim),
                                            requires_grad=True))
            self.numel_weights += self.input_dim * out_dim * in_dim

    def forward_given_params(self, x, weights, biases):
        """

        Parameters
        ----------
        x: batch_size x num_vars
        weights: List
            ith list contains weights for ith MLP
        biases: List
            ith list contains biases for ith MLP
        Returns
        -------
        out : batch_size x num_vars * num_params
            the parameters of each variable conditional
        """

        for k in range(self.hidden_num + 1):
            # apply affine operator
            if k == 0:
                adj = self.adjacency.unsqueeze(0)
                x = torch.einsum("tij,ljt,bj->bti", weights[k], adj, x) + biases[k]
            else:
                x = torch.einsum("tij,btj->bti", weights[k], x) + biases[k]

            # apply non-linearity
            if k != self.hidden_num:
                if self.nonlinear == "leaky-relu":
                    x = F.leaky_relu(x)
                else:
                    x = torch.sigmoid(x)

        return torch.unbind(x, 1)

    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        return compute_A_phi(self, norm=self.norm_prod, square=self.square_prod)

    def reset_params(self):
        with torch.no_grad():
            for node in range(self.input_dim):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(w, 
                                    gain=nn.init.calculate_gain('leaky_relu'))
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

    def get_parameters(self, mode="wbx"):
        """Will get only parameters with requires_grad == True

        Parameters
        ----------
        mode: str
            w=weights, b=biases, x=extra_params (order is irrelevant)
        Returns
        -------
        out : tuple
            corresponding dicts of parameters
        """

        params = []

        if 'w' in mode:
            weights = []
            for w in self.weights:
                weights.append(w)
            params.append(weights)
        if 'b'in mode:
            biases = []
            for j, b in enumerate(self.biases):
                biases.append(b)
            params.append(biases)

        if 'x' in mode:
            extra_params = []
            for ep in self.extra_params:
                if ep.requires_grad:
                    extra_params.append(ep)
            params.append(extra_params)

        return tuple(params)

    def set_parameters(self, params, mode="wbx"):
        """Will set only parameters with requires_grad == True

        Parameters
        ----------
        params : tuple of parameter lists to set,
            the order should be coherent with `get_parameters`
        mode : str
            w=weights, b=biases, x=extra_params (order is irrelevant)
        """

        with torch.no_grad():
            k = 0
            if 'w' in mode:
                for i, w in enumerate(self.weights):
                    w.copy_(params[k][i])
                k += 1

            if 'b' in mode:
                for i, b in enumerate(self.biases):
                    b.copy_(params[k][i])
                k += 1

            if 'x' in mode and len(self.extra_params) > 0:
                for i, ep in enumerate(self.extra_params):
                    if ep.requires_grad:
                        ep.copy_(params[k][i])
                k += 1

    def get_grad_norm(self, mode="wbx"):
        """Will get only parameters with requires_grad == True

        Parameters
        ----------
        mode: str
            w=weights, b=biases, x=extra_params (order is irrelevant)
        Returns
        -------
        out : tuple
            corresponding dicts of parameters
        """

        grad_norm = torch.zeros(1)

        if 'w' in mode:
            for w in self.weights:
                grad_norm += torch.sum(w.grad ** 2)

        if 'b'in mode:
            for j, b in enumerate(self.biases):
                grad_norm += torch.sum(b.grad ** 2)

        if 'x' in mode:
            for ep in self.extra_params:
                if ep.requires_grad:
                    grad_norm += torch.sum(ep.grad ** 2)

        return torch.sqrt(grad_norm)

    def get_distribution(self, density_params):
        raise NotImplementedError


class LearnableModel(BaseModel):
    """Class for other learnable Models, disable instantiation."""

    def __init__(self,
                 input_dim,
                 hidden_num,
                 hidden_dim,
                 output_dim,
                 nonlinear="leaky-relu",
                 norm_prod='path',
                 square_prod=False):
        super(LearnableModel, self).__init__(input_dim=input_dim,
                                             hidden_num=hidden_num,
                                             hidden_dim=hidden_dim,
                                             output_dim=output_dim,
                                             nonlinear=nonlinear,
                                             norm_prod=norm_prod,
                                             square_prod=square_prod)
        self.reset_params()

    def compute_log_likelihood(self, x, weights, biases, extra_params,
                               detach=False):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution
            only if the DAGness constraint on the mask is satisfied.
            Otherwise the joint does not integrate to one.

        Parameters
        ----------
        x: tuple
            (batch_size, input_dim)
        weights: list of tensor
            that are coherent with self.weights
        biases: list of tensor
            that are coherent with self.biases
        Returns
        -------
        (batch_size, input_dim) log-likelihoods
        """
        density_params = self.forward_given_params(x, weights, biases)

        if len(extra_params) != 0:
            extra_params = self.transform_extra_params(self.extra_params)
        log_probs = []
        for i in range(self.input_dim):
            density_param = list(torch.unbind(density_params[i], 1))
            if len(extra_params) != 0:
                density_param.extend(list(torch.unbind(extra_params[i], 0)))
            conditional = self.get_distribution(density_param)
            x_d = x[:, i].detach() if detach else x[:, i]
            log_probs.append(conditional.log_prob(x_d).unsqueeze(1))

        return torch.cat(log_probs, 1)

    def get_distribution(self, dp):
        raise NotImplementedError

    def transform_extra_params(self, extra_params):
        raise NotImplementedError


class NonlinearGauss(LearnableModel):
    """Class of learnable models based on NonlinearGauss

    Parameters
    ----------
    input_dim : int
        number of features
    hidden_num : int
        number of hidden layers
    hidden_dim : int
        dimension of per hidden layer
    output_dim : int, 2
    nonlinear : str
        Nonlinear activation function
    norm_prod : str, 'path'
    square_prod : bool, default False
        whether use square_prod
    """

    def __init__(self,
                 input_dim,
                 hidden_num,
                 hidden_dim,
                 output_dim,
                 nonlinear="leaky-relu",
                 norm_prod='path',
                 square_prod=False):
        super(NonlinearGauss, self).__init__(input_dim=input_dim,
                                             hidden_num=hidden_num,
                                             hidden_dim=hidden_dim,
                                             output_dim=output_dim,
                                             nonlinear=nonlinear,
                                             norm_prod=norm_prod,
                                             square_prod=square_prod)


    def get_distribution(self, dp):
        return distributions.normal.Normal(dp[0], torch.exp(dp[1]))


class NonlinearGaussANM(LearnableModel):
    """Class of learnable models based on NonlinearGaussANM

    Parameters
    ----------
    input_dim : int
        number of features
    hidden_num : int
        number of hidden layers
    hidden_dim : int
        dimension of per hidden layer
    output_dim : int, 2
    nonlinear : str
        Nonlinear activation function
    norm_prod : str, 'path'
    square_prod : bool, default False
    """

    def __init__(self,
                 input_dim,
                 hidden_num,
                 hidden_dim,
                 output_dim,
                 nonlinear="leaky-relu",
                 norm_prod='path',
                 square_prod=False):
        super(NonlinearGaussANM, self).__init__(input_dim=input_dim,
                                                hidden_num=hidden_num,
                                                hidden_dim=hidden_dim,
                                                output_dim=output_dim,
                                                nonlinear=nonlinear,
                                                norm_prod=norm_prod,
                                                square_prod=square_prod)
        # extra parameters are log_std
        extra_params = np.ones((self.input_dim,))
        np.random.shuffle(extra_params)
        # each element in the list represents a variable,
        # the size of the element is the number of extra_params per var
        self.extra_params = nn.ParameterList()
        for extra_param in extra_params:
            self.extra_params.append(
                nn.Parameter(
                    torch.tensor(
                        np.log(extra_param).reshape(1)).type(torch.Tensor),
                    requires_grad=True))

    def get_distribution(self, dp):
        return distributions.normal.Normal(dp[0], dp[1])

    def transform_extra_params(self, extra_params):
        transformed_extra_params = []
        for extra_param in extra_params:
            transformed_extra_params.append(torch.exp(extra_param))
        return transformed_extra_params  # returns std_dev
