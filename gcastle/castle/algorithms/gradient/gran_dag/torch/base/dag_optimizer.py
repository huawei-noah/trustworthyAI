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

from castle.common.validator import transfer_to_device


def compute_constraint(model, w_adj):
    assert (w_adj >= 0).detach().cpu().numpy().all()
    h = torch.trace(torch.matrix_exp(w_adj)) - model.input_dim
    return h


def is_acyclic(adjacency, device=None):
    """
    Whether the adjacency matrix is a acyclic graph.
    """
    prod = np.eye(adjacency.shape[0])
    adjacency, prod = transfer_to_device(adjacency, prod, device=device)
    for _ in range(1, adjacency.shape[0] + 1):
        prod = torch.matmul(adjacency, prod)
        if torch.trace(prod) != 0:
            return False
    return True


def compute_A_phi(model, norm="none", square=False):
    """
    compute matrix A consisting of products of NN weights
    """
    weights = model.get_parameters(mode='w')[0]
    prod = torch.eye(model.input_dim)
    if norm != "none":
        prod_norm = torch.eye(model.input_dim)
    for i, w in enumerate(weights):
        if square:
            w = w ** 2
        else:
            w = torch.abs(w)
        if i == 0:
            prod = torch.einsum("tij,ljt,jk->tik", w,
                                model.adjacency.unsqueeze(0), prod)
            if norm != "none":
                tmp = 1. - torch.eye(model.input_dim).unsqueeze(0)
                prod_norm = torch.einsum("tij,ljt,jk->tik",
                                         torch.ones_like(w).detach(), tmp,
                                         prod_norm)
        else:
            prod = torch.einsum("tij,tjk->tik", w, prod)
            if norm != "none":
                prod_norm = torch.einsum("tij,tjk->tik",
                                         torch.ones_like(w).detach(),
                                         prod_norm)

    # sum over density parameter axis
    prod = torch.sum(prod, 1)
    if norm == "paths":
        prod_norm = torch.sum(prod_norm, 1)
        denominator = prod_norm + torch.eye(model.input_dim)  # avoid / 0 on diagonal
        return (prod / denominator).t()
    elif norm == "none":
        return prod.t()
    else:
        raise NotImplementedError


def compute_jacobian_avg(model, data_manager, batch_size):
    """
    compute the average Jacobian of learned model
    """
    jac_avg = torch.zeros(model.input_dim, model.input_dim)

    # sample
    x, do_mask = data_manager.sample(batch_size)
    x.requires_grad = True

    # compute loss
    weights, biases, extra_params = model.get_parameters(mode="wbx")
    log_probs = model.compute_log_likelihood(x, weights, biases, extra_params,
                                             detach=True)
    log_probs = torch.unbind(log_probs, 1)

    # compute jacobian of the loss
    for i in range(model.input_dim):
        tmp = torch.autograd.grad(log_probs[i], x, retain_graph=True,
                                  grad_outputs=torch.ones(batch_size))[0]
        jac_avg[i, :] = torch.abs(tmp).mean(0)

    return jac_avg
