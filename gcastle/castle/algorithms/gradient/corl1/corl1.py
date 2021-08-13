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
import logging
import argparse
import platform
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from .data_loader import DataGenerator_read_data
from .models import Actor
from .rewards import get_Reward
from .helpers.tf_utils import set_seed
from .helpers.analyze_utils import graph_prunned_by_coef, \
    graph_prunned_by_coef_2nd, cover_rate, from_order_to_graph

from castle.common import BaseLearner, Tensor
from castle.metrics import MetricsDAG


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class CORL1(BaseLearner):
    """
    CORL1 Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Parameters
    ----------
    encoder_type: str
        type of encoder used
    hidden_dim: int
        actor LSTM num_neurons
    num_heads: int
        actor input embedding
    num_stacks: int
        actor LSTM num_neurons
    residual: bool
        whether to use residual for gat encoder
    decoder_type: str
        type of decoder used
    decoder_activation: str
        activation for decoder, Choose from: 'tanh', 'relu', 'none'
    decoder_hidden_dim: int
        hidden dimension for decoder
    use_bias: bool
        Whether to add bias term when calculating decoder logits
    use_bias_constant: bool
        Whether to add bias term as CONSTANT when calculating decoder logits
    bias_initial_value: float
        Initial value for bias term when calculating decoder logits
    batch_size: int
        batch size for training
    input_dimension: int
        dimension of reshaped vector
    normalize: bool
        whether the inputdata shall be normalized
    transpose: bool
        whether the true graph needs transposed
    parral: bool
        whether multi-process to cal reward
    median_flag: bool
        whether the median needed in GPR
    restore_model_path: str
        data path for restore data
    score_type: str
        score functions
    reg_type: str
        regressor type (in combination wth score_type)
    lambda_iter_num: int
        how often to update lambdas
    lambda_flag_default: bool
        with set lambda parameters; true with default strategy and ignore input bounds
    score_bd_tight: bool
        if bound is tight, then simply use a fixed value, rather than the adaptive one
    lambda1_update: float
        increasing additive lambda1
    lambda2_update: float
        increasing  multiplying lambda2
    score_lower: float
        lower bound on lambda1
    score_upper: float
        upper bound on lambda1
    lambda2_lower: float
        lower bound on lambda2
    lambda2_upper: float
        upper bound on lambda2
    med_w: float
        specify median
    seed: int
        seed
    nb_epoch: int
        nb epoch
    lr1_start: float
        actor learning rate
    lr1_decay_step: int
        lr1 decay step
    lr1_decay_rate: float
        lr1 decay rate
    alpha: float
        update factor moving average baseline
    init_baseline: float
        initial baseline - REINFORCE
    temperature: float
        pointer_net initial temperature
    C: float
        pointer_net tan clipping
    l1_graph_reg: float
        L1 graph regularization to encourage sparsity
    lr3: float
        pointer_net tan clipping
    gpr_alpha: float
        gpr_alpha

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix.

    Examples
    --------
    >>> from castle.algorithms import CORL1
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> true_dag, X = load_dataset(name='iid_test')
    >>> n = CORL1()
    >>> n.learn(X)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self, encoder_type='TransformerEncoder', 
                 hidden_dim=64, 
                 num_heads=16, 
                 num_stacks=3, 
                 residual=False, 
                 decoder_type='PointerDecoder', 
                 decoder_activation='tanh', 
                 decoder_hidden_dim=16, 
                 use_bias=False, 
                 use_bias_constant=False, 
                 bias_initial_value=False, 
                 batch_size=64, 
                 input_dimension=64, 
                 normalize=False, 
                 transpose=False, 
                 parral=False, 
                 median_flag=False, 
                 restore_model_path='data', 
                 score_type='BIC', 
                 reg_type='LR', 
                 lambda_iter_num=1000, 
                 lambda_flag_default=False, 
                 score_bd_tight=False, 
                 lambda1_update=1, 
                 lambda2_update=10, 
                 score_lower=0.0, 
                 score_upper=0.0, 
                 lambda2_lower=-1, 
                 lambda2_upper=-1, 
                 med_w=1, 
                 seed=8, 
                 nb_epoch=20000, 
                 lr1_start=0.001, 
                 lr1_decay_step=5000, 
                 lr1_decay_rate=0.96, 
                 alpha=0.99, 
                 init_baseline=-1.0, 
                 temperature=3.0, 
                 C=10.0, 
                 l1_graph_reg=0.0, 
                 lr3=0.0001, 
                 gpr_alpha=1.0):

        super().__init__()

        parser = argparse.ArgumentParser(description='Configuration')
        self.config = parser.parse_args(args=[])
        self.config.encoder_type = encoder_type
        self.config.hidden_dim = hidden_dim
        self.config.num_heads = num_heads
        self.config.num_stacks = num_stacks
        self.config.residual = residual
        self.config.decoder_type = decoder_type
        self.config.decoder_activation = decoder_activation
        self.config.decoder_hidden_dim = decoder_hidden_dim
        self.config.use_bias = use_bias
        self.config.use_bias_constant = use_bias_constant
        self.config.bias_initial_value = bias_initial_value
        self.config.batch_size = batch_size
        self.config.input_dimension = input_dimension
        self.config.normalize = normalize
        self.config.transpose = transpose
        self.config.parral = parral
        self.config.median_flag = median_flag
        self.config.restore_model_path = restore_model_path
        self.config.score_type = score_type
        self.config.reg_type = reg_type
        self.config.lambda_iter_num = lambda_iter_num
        self.config.lambda_flag_default = lambda_flag_default
        self.config.score_bd_tight = score_bd_tight
        self.config.lambda1_update = lambda1_update
        self.config.lambda2_update = lambda2_update
        self.config.score_lower = score_lower
        self.config.score_upper = score_upper
        self.config.lambda2_lower = lambda2_lower
        self.config.lambda2_upper = lambda2_upper
        self.config.med_w = med_w
        self.config.seed = seed
        self.config.nb_epoch = nb_epoch
        self.config.lr1_start = lr1_start
        self.config.lr1_decay_step = lr1_decay_step
        self.config.lr1_decay_rate = lr1_decay_rate
        self.config.alpha = alpha
        self.config.init_baseline = init_baseline
        self.config.temperature = temperature
        self.config.C = C
        self.config.l1_graph_reg = l1_graph_reg
        self.config.lr3 = lr3
        self.config.gpr_alpha = gpr_alpha

    def learn(self, data, dag=None):
        """
        Set up and run the CORL1 algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        """
        config = self.config
        if dag is not None:
            config.dag = dag

        if isinstance(data, np.ndarray):
            X = data
        elif isinstance(data, Tensor):
            X = data.data
        else:
            raise TypeError('The type of data must be '
                            'Tensor or numpy.ndarray, but got {}'
                            .format(type(data)))
        
        config.data_size = X.shape[0]
        config.max_length = X.shape[1]

        causal_matrix = self._rl(X, config)
        self.causal_matrix = causal_matrix

    def _rl(self, X, config):
        """
        Starting model of CORL1.

        Parameters
        ----------
        X: numpy.ndarray
            The numpy.ndarray format data you want to learn.
        config: dict
            The parameters dict for corl1.
        """

        set_seed(config.seed)

        logging.info('Python version is {}'.format(platform.python_version()))

        # input data
        solution_path_mask = None
        if hasattr(config, 'dag'):
            training_set = DataGenerator_read_data(
                X, config.dag, solution_path_mask, config.normalize, config.transpose)
        else:
            training_set = DataGenerator_read_data(
                X, None, solution_path_mask, config.normalize, config.transpose)
        input_data = training_set.inputdata[:config.data_size,:]

        # set penalty weights
        score_type = config.score_type
        reg_type = config.reg_type

        actor = Actor(config)
        callreward = get_Reward(actor.batch_size, config.max_length, 
                                config.parral, actor.input_dimension,
                                input_data, score_type, reg_type, 
                                config.gpr_alpha, config.med_w, 
                                config.median_flag, config.l1_graph_reg, False)
        logging.info('Finished creating training dataset, actor model and reward class')

        logging.info('Starting session...')
        sess_config = tf.ConfigProto(log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            logging.info('Shape of actor.input: {}'.format(sess.run(tf.shape(actor.input_))))

            reward_max_per_batch = []
            reward_mean_per_batch = []
            max_rewards = []
            max_reward = float('-inf')

            max_sum = 0

            logging.info('Starting training.')
            loss1_s, loss_2s = [], []
            
            for i in (range(1, config.nb_epoch+1)):
                
                # if i%1000 == 0 or i%1000 == 1:
                #     logging.info('time log_3')

                input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
                positions, i_list, s0_list, s1_list = sess.run([actor.positions, actor.i_list,
                        actor.s0_list, actor.s1_list],feed_dict={actor.input_: input_batch})

                samples = []
                action_mask_s = []
                for m in range(positions.shape[0]):
                    zero_matrix = from_order_to_graph(positions[m])

                    action_mask = np.zeros(actor.max_length)
                    for po in positions[m]:
                        action_mask_s.append(action_mask.copy())
                        action_mask += np.eye(actor.max_length)[po]

                    samples.append(zero_matrix)
                    temp_sum = cover_rate(zero_matrix, training_set.true_graph.T)
                    if temp_sum > max_sum:
                        max_sum = temp_sum
                        if i == 1 or i % 500 == 0:
                            logging.info('[iter {}] [Batch {}_th] The optimal nodes order cover true graph {}/{}!'.format(i,m,max_sum,actor.max_length*actor.max_length))
                
                # if i % 1000 == 0:
                #     logging.info('time log_1')

                graphs_feed = np.stack(samples)
                action_mask_s = np.stack(action_mask_s)
                reward_feed = callreward.cal_rewards(graphs_feed, positions)
                
                # if i % 1000 == 0:
                #     logging.info('time log_2')

                max_reward_batch = -float('inf')
                reward_list, normal_batch_reward = [], []
                for nu, (reward_, reward_list_) in enumerate(reward_feed):
                    reward_list.append(reward_list_)
                    normalized_reward = -reward_
                    normal_batch_reward.append(normalized_reward)
                    if normalized_reward > max_reward_batch:
                        max_reward_batch = normalized_reward
                if max_reward < max_reward_batch:
                    max_reward = max_reward_batch

                reward_list = - np.stack(reward_list)
                normal_batch_reward = np.stack(normal_batch_reward)

                G = 0
                td_target = []
                for r in np.transpose(reward_list, [1, 0])[::-1]:
                    G = r + actor.gamma * G
                    td_target.append(G)

                feed = {actor.input_: input_batch, actor.reward_: normal_batch_reward,
                        actor.reward_list: reward_list, actor.target_values_: td_target[::-1],
                    actor.i_list_ev:np.transpose(i_list,[1,0,2])[:-1], actor.i_list_ta:np.transpose(i_list,[1,0,2])[1:],
                    actor.prev_state_0: s0_list.reshape((-1, actor.input_dimension)),
                    actor.prev_state_1: s1_list.reshape((-1, actor.input_dimension)),
                    actor.prev_input: i_list.reshape((-1, actor.input_dimension)),
                    actor.position: positions.reshape(actor.batch_size*actor.max_length),
                    actor.action_mask_: action_mask_s.reshape((-1,actor.max_length))}

                soft_replacement, log_softmax, train_step1, loss1, loss2, train_step2 = \
                    sess.run([actor.soft_replacement, actor.log_softmax,
                    actor.train_step1, actor.loss1, actor.loss2,actor.train_step2], feed_dict=feed)

                loss1_s.append(loss1)
                loss_2s.append(loss2)

                reward_max_per_batch.append(max_reward_batch)
                reward_mean_per_batch.append(np.mean(normal_batch_reward))
                max_rewards.append(max_reward)

                # logging
                if i == 1 or i % 500 == 0:
                    logging.info('[iter {}] max_reward: {:.4}, max_reward_batch: {:.4}'.format(i, max_reward, max_reward_batch))

                if i == 1 or i % config.lambda_iter_num == 0:
                    ls_kv = callreward.update_all_scores()

                    score_min, graph_int_key = ls_kv[0][1][0], ls_kv[0][0]
                    logging.info('[iter {}] score_min {:.4}'.format(i, score_min*1.0))
                    graph_batch = from_order_to_graph(graph_int_key)

                    if reg_type == 'LR':
                        graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, input_data))
                    elif reg_type == 'QR':
                        graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))
                    
                    if hasattr(config, 'dag'):
                        met = MetricsDAG(graph_batch.T, training_set.true_graph)
                        met2 = MetricsDAG(graph_batch_pruned.T, training_set.true_graph)
                        acc_est = met.metrics
                        acc_est2 = met2.metrics

                        fdr, tpr, fpr, shd, nnz = \
                            acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], \
                            acc_est['shd'], acc_est['nnz']
                        fdr2, tpr2, fpr2, shd2, nnz2 = \
                            acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], \
                            acc_est2['shd'], acc_est2['nnz']
                        
                        logging.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                        logging.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))

            plt.figure(1)
            plt.plot(reward_max_per_batch, label='max reward per batch')
            plt.plot(reward_mean_per_batch, label='mean reward per batch')
            plt.plot(max_rewards, label='max reward')
            plt.legend()
            plt.savefig('reward_batch_average.png')
            plt.close()

            logging.info('Training COMPLETED!')
        return graph_batch_pruned.T
