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
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import pdist


def BIC_input_graph(X, g, reg_type='LR', score_type='BIC'):
    """cal BIC score for given graph"""

    RSS_ls = []

    n, d = X.shape 

    if reg_type in ('LR', 'QR'):
        reg = LinearRegression()
    else:
        reg =GaussianProcessRegressor()

    poly = PolynomialFeatures()

    for i in range(d):
        y_ = X[:, [i]]
        inds_x = list(np.abs(g[i])>0.1)

        if np.sum(inds_x) < 0.1: 
            y_pred = np.mean(y_)
        else:
            X_ = X[:, inds_x]
            if reg_type == 'QR':              
                X_ = poly.fit_transform(X_)[:, 1:] 
            elif reg_type == 'GPR':                
                med_w = np.median(pdist(X_, 'euclidean'))
                X_ = X_ / med_w
            reg.fit(X_, y_)
            y_pred = reg.predict(X_)
        RSSi = np.sum(np.square(y_ - y_pred))

        if reg_type == 'GPR':
            RSS_ls.append(RSSi+1.0)
        else:
            RSS_ls.append(RSSi)

    if score_type == 'BIC':
        return np.log(np.sum(RSS_ls)/n+1e-8) 
    elif score_type == 'BIC_different_var':
        return np.sum(np.log(np.array(RSS_ls)/n)+1e-8) 
    
    
def BIC_lambdas(X, gl=None, gu=None, gtrue=None, reg_type='LR', score_type='BIC'):
    """
    :param X: dataset
    :param gl: input graph to get score lower bound
    :param gu: input graph to get score upper bound
    :param gtrue: input true graph
    :param reg_type:
    :param score_type:
    :return: score lower bound, score upper bound, true score (only for monitoring)
    """
        
    n, d = X.shape

    if score_type == 'BIC':
        bic_penalty = np.log(n) / (n*d)
    elif score_type == 'BIC_different_var':
        bic_penalty = np.log(n) / n
    
    # default gl for BIC score: complete graph (except digonals)
    if gl is None:
        g_ones= np.ones((d,d))
        for i in range(d):
            g_ones[i, i] = 0
        gl = g_ones

    # default gu for BIC score: empty graph
    if gu is None:
        gu = np.zeros((d, d))

    sl = BIC_input_graph(X, gl, reg_type, score_type)
    su = BIC_input_graph(X, gu, reg_type, score_type) 

    if gtrue is None:
        strue = sl - 10
    else:
        print(BIC_input_graph(X, gtrue, reg_type, score_type))
        print(gtrue)
        print(bic_penalty)
        strue = BIC_input_graph(X, gtrue, reg_type, score_type) + np.sum(gtrue) * bic_penalty
    
    return sl, su, strue

