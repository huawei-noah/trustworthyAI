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
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky, cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GPR_mine:
    def __init__(self, optimize=False):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 1, "sigma_f": 1}
        self.optimize = optimize
        self.alpha = 1e-10
        self.m = None

    def fit(self, y, median, p_eu):
        self.train_y = np.asarray(y)
        K = self.kernel(median, p_eu)
        np.fill_diagonal(K, 1)
        self.K_trans = K.copy()
        K[np.diag_indices_from(K)] += self.alpha
        #self.KK = K.copy()

        self.L_ = cholesky(K, lower=True)  # Line 2
        # self.L_ changed, self._K_inv needs to be recomputed
        self._K_inv = None
        self.alpha_ = cho_solve((self.L_, True), self.train_y)  # Line 3
        self.is_fit = True

    def predict(self, return_std=False):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        K_trans = self.K_trans
        y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        if return_std == False:
            return y_mean
        else:
            raise ('To cal std')

    def kernel(self, median, p_eu):
        p_eu_nor = p_eu / median
        K = np.exp(-0.5 * p_eu_nor)
        K = squareform(K)
        return K


class get_Reward(object):

    def __init__(self, batch_num, maxlen, parral, dim, inputdata, score_type='BIC',
                 reg_type='LR', alpha=1.0, med_w=1.0, median_flag=False, l1_graph_reg=0.0, verbose_flag=True):
        self.batch_num = batch_num
        self.maxlen = maxlen # =d: number of vars
        self.dim = dim
        #self.baseint = 2**maxlen
        self.alpha = alpha
        self.med_w = med_w
        self.med_w_flag = median_flag
        self.d = {} # store results
        self.d_RSS = [{} for _ in range(maxlen)]  # store RSS for reuse
        self.inputdata = inputdata.astype(np.float32)

        self.n_samples = inputdata.shape[0]
        self.l1_graph_reg = l1_graph_reg 
        self.verbose = verbose_flag
        self.bic_penalty = np.log(inputdata.shape[0])/inputdata.shape[0]

        if score_type not in ('BIC', 'BIC_different_var'):
            raise ValueError('Reward type not supported.')
        if reg_type not in ('LR', 'QR', 'GPR','GPR_learnable'):
            raise ValueError('Reg type not supported')
        self.score_type = score_type
        self.reg_type = reg_type

        self.poly = PolynomialFeatures()

        if self.reg_type == 'GPR_learnable':
            self.kernel_learnable = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                          + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e+1))
        elif reg_type=='LR':
            self.ones = np.ones((inputdata.shape[0], 1), dtype=np.float32)
            X = np.hstack((self.inputdata, self.ones))
            self.X = X
            self.XtX = X.T.dot(X)
        elif reg_type=='GPR':
            self.gpr = GPR_mine()
            m = inputdata.shape[0]
            self.gpr.m = m
            dist_matrix = []
            for i in range(m):
                for j in range(i + 1, m):
                    dist_matrix.append((inputdata[i] - inputdata[j]) ** 2)
            self.dist_matrix = np.array(dist_matrix)

    def cal_rewards(self, graphs, positions=None, ture_flag=False):
        rewards_batches = []
        if not ture_flag:
            for graphi, position in zip(graphs, positions):
                reward_ = self.calculate_reward_single_graph(graphi, position=position, ture_flag=ture_flag)
                rewards_batches.append(reward_)
        else:
            for graphi in graphs:
                reward_ = self.calculate_reward_single_graph(graphi, ture_flag=ture_flag)
                rewards_batches.append(reward_)
        return rewards_batches

    ####### regression
    def calculate_yerr(self, X_train, y_train, XtX, Xty):
        if self.reg_type == 'LR':
            return self.calculate_LR(X_train, y_train, XtX, Xty)
        elif self.reg_type == 'QR':
            return self.calculate_QR(X_train, y_train)
        elif self.reg_type == 'GPR':
            return self.calculate_GPR(X_train, y_train, XtX, Xty)
        elif self.reg_type == 'GPR_learnable':
            return self.calculate_GPR_learnable(X_train, y_train)
        else:
            assert False, 'Regressor not supported'

    # faster than LinearRegression() from sklearn
    #numba
    def calculate_LR(self, X_train, y_train, XtX, Xty):
        theta = np.linalg.solve(XtX, Xty)
        y_pre = X_train.dot(theta)
        y_err = y_pre - y_train
        return y_err

    def calculate_QR(self, X_train, y_train):
        X_train = self.poly.fit_transform(X_train)[:,1:]
        return self.calculate_LR(X_train, y_train)

    def calculate_GPR(self, X_train, y_train, K1, K2):
        p_eu = K1   #TODO our K1 don't sqrt
        med_w = np.median(p_eu)
        self.gpr.fit(y_train, med_w, p_eu)
        pre = self.gpr.predict()
        return y_train - pre

    def calculate_GPR_learnable(self, X_train, y_train):
        gpr = GPR(kernel=self.kernel_learnable, alpha= 0.0).fit(X_train, y_train)
        return y_train.reshape(-1, 1) - gpr.predict(X_train).reshape(-1, 1)

    def calculate_reward_single_graph(self, graph_batch, position=None, ture_flag=False):
        if not ture_flag:
            graph_to_int2 = list(np.int32(position))
            graph_batch_to_tuple = tuple(graph_to_int2)

            if graph_batch_to_tuple in self.d:
                graph_score = self.d[graph_batch_to_tuple]
                reward = graph_score[0]
                # print('No cal , query dict d directly!')
                return reward, np.array(graph_score[1])

        RSS_ls = []
        for i in range(self.maxlen):
            RSSi = self.cal_RSSi(i, graph_batch)
            RSS_ls.append(RSSi)

        RSS_ls = np.array(RSS_ls)
        if self.reg_type == 'GPR' or self.reg_type == 'GPR_learnable':
            reward_list = RSS_ls[position] / self.n_samples
        else:
            reward_list = RSS_ls[position] / self.n_samples

        if self.score_type == 'BIC':
            BIC = np.log(np.sum(RSS_ls)/self.n_samples+1e-8) 
                 # + np.sum(graph_batch)*self.bic_penalty/self.maxlen 
        elif self.score_type == 'BIC_different_var':
            BIC = np.sum(np.log(np.array(RSS_ls)/self.n_samples+1e-8)) 
                # + np.sum(graph_batch)*self.bic_penalt

        if not ture_flag:
            self.d[graph_batch_to_tuple] = (BIC, reward_list)
        return BIC, np.array(reward_list)

    def cal_RSSi(self, i, graph_batch):
        col = graph_batch[i]
        str_col = str(col)
        if str_col in self.d_RSS[i]:
            RSSi = self.d_RSS[i][str_col]
            return RSSi

        if np.sum(col) < 0.1:
            y_err = self.inputdata[:, i]
            y_err = y_err - np.mean(y_err)
        else:
            cols_TrueFalse = col > 0.5
            if self.reg_type == 'LR':
                cols_TrueFalse = np.append(cols_TrueFalse, True)
                X_train = self.X[:, cols_TrueFalse]
                y_train = self.X[:, i]

                XtX = self.XtX[:, cols_TrueFalse][cols_TrueFalse,:]
                Xty = self.XtX[:, i][cols_TrueFalse]

                y_err = self.calculate_yerr(X_train, y_train, XtX, Xty)

            elif self.reg_type == 'GPR':
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                p_eu = pdist(X_train, 'sqeuclidean')
                if self.med_w_flag:
                    self.med_w = np.median(p_eu)
                train_y = np.asarray(y_train)
                #print('med_w:',self.med_w)
                p_eu_nor = p_eu / self.med_w
                K = np.exp(-0.5 * p_eu_nor)
                K = squareform(K)

                np.fill_diagonal(K, 1)
                K_trans = K.copy()
                K[np.diag_indices_from(K)] += self.alpha #1e-10
                L_ = cholesky(K, lower=True)  # Line 2
                alpha_ = cho_solve((L_, True),train_y)  # Line 3
    
                y_mean = K_trans.dot(alpha_)  # Line 4 (y_mean = f_star)
                y_err = y_train - y_mean

            elif self.reg_type == 'GPR_learnable':
                X_train = self.inputdata[:, cols_TrueFalse]
                y_train = self.inputdata[:, i]
                y_err = self.calculate_yerr(X_train, y_train,X_train, y_train)

        RSSi = np.sum(np.square(y_err))
        self.d_RSS[i][str_col] = RSSi

        return RSSi

    #### helper
    def penalized_score(self, score_cyc, lambda1=1, lambda2=1):
        score, cyc = score_cyc
        return score + lambda1*np.float(cyc>1e-5) + lambda2*cyc
    
    def update_scores(self, score_cycs):
        ls = []
        for score_cyc in score_cycs:
            ls.append(score_cyc)
        return ls
    
    def update_all_scores(self):
        score_cycs = list(self.d.items())
        ls = []
        for graph_int, score_l in score_cycs:
            ls.append((graph_int, (score_l[0], score_l[-1])))
        return sorted(ls, key=lambda x: x[1][0])
