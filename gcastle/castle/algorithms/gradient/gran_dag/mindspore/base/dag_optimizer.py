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
from mindspore import Tensor as MsTensor, ops, dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as msnp


class GradNetWrtX(nn.Cell):
    """
    Calculates the derivative of output to the input

    Parameters
    ----------
    network: instantiating model object
        'NonLinGaussANM' or 'NonLinGauss'
    """
    def __init__(self, network):
        super(GradNetWrtX, self).__init__(auto_prefix=False)
        self.network = network
        self.grad_op = ops.GradOperation(sens_param=True)
        self.flag = MsTensor(True, dtype=mstype.bool_)

    def construct(self, x, grad_output):
        """
        If you want to calculate the derivative of a single output
        (for example, output[0][0]) to the input,
        you can set the scaling value of the corresponding position to 1
        and set other values to 0.

        Parameters
        ----------
        x: mindspore.Tensor
        grad_output: mindspore.Tensor, have same shape with x

        Returns
        -------
        gout: gradient value, have same shape with x
        """
        gradient_function = self.grad_op(self.network)
        gout = gradient_function(x, self.flag, grad_output)
        return gout


def is_acyclic(adjacency):
    """
    Whether the adjacency matrix is a acyclic graph.
    """
    prod = np.eye(adjacency.shape[0])
    for _ in range(1, adjacency.shape[0] + 1):
        prod = np.matmul(adjacency.asnumpy(), prod)
        if np.trace(prod) != 0:
            return False
    return True


def compute_A_phi(model, norm="none", square=False):
    """
    compute matrix A consisting of products of NN weights

    Parameters
    ----------
    model: instantiating model objects
        'NonLinGaussANM' or 'NonLinGauss'
    norm: str, default 'none'
        use norm of product of paths, 'none' or 'paths'
        'paths': use norm, 'none': with no norm
    square: bool, default False
        use squared product of paths
    """
    weights = model.get_parameters(mode='w')[0]
    prod = msnp.eye(model.input_dim, dtype=mstype.float32)
    if norm == "paths":
        prod_norm = msnp.eye(model.input_dim)
    for i, w in enumerate(weights):
        if square:
            w = w ** 2
        else:
            w = ops.absolute(w)
        if i == 0:
            tmp_adj = ops.expand_dims(model.adjacency.transpose(), 1)
            ein_one = ops.mul(w, tmp_adj)
            prod = ops.matmul(ein_one, prod)
            if norm == "paths":
                tmp_adj = 1. - msnp.eye(model.input_dim, dtype=mstype.float32)
                tmp_adj = ops.expand_dims(tmp_adj.transpose(), 1)
                ein_two = ops.mul(ops.ones_like(w), tmp_adj)
                prod_norm = ops.matmul(ein_two, prod_norm)
        else:
            prod = ops.matmul(w, prod)
            if norm == "paths":
                prod_norm = ops.matmul(ops.ones_like(w), prod_norm)

    # sum over density parameter axis
    prod = ops.reduce_sum(prod, 1)
    if norm == "paths":
        prod_norm = ops.reduce_sum(prod_norm, 1)
        # avoid / 0 on diagonal
        denominator = prod_norm + msnp.eye(model.input_dim,
                                           dtype=mstype.float32)
        return (prod / denominator).transpose()
    elif norm == "none":
        return prod.transpose()
    else:
        raise NotImplementedError


def compute_jacobian_avg(model, data_manager, batch_size):
    """
    compute the average Jacobian of learned model
    """
    jac_avg = msnp.zeros((model.input_dim, model.input_dim),
                         dtype=mstype.float32)

    # sample
    x, _ = data_manager.sample(batch_size)
    model.set_train(False)

    # compute jacobian of the loss
    for k in range(model.input_dim):
        grad_output = msnp.zeros(shape=x.shape, dtype=mstype.float32)
        grad_output[:, k] = 1
        tmp_grad = GradNetWrtX(model)(x, grad_output)
        jac_avg[k, :] = ops.reduce_mean(ops.absolute(tmp_grad), 0)

    return jac_avg
