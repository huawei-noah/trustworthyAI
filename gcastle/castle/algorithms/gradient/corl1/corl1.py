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
import platform
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from loguru import logger

from .data_loader import DataGenerator_read_data
from .models import Actor
from .rewards import get_Reward
from .helpers.config_graph import get_config
from .helpers.tf_utils import set_seed
from .helpers.analyze_utils import graph_prunned_by_coef, \
    graph_prunned_by_coef_2nd, cover_rate, from_order_to_graph

from castle.common import BaseLearner, Tensor
from castle.metrics import MetricsDAG


class CORL1(BaseLearner):
    """
    CORL1 Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

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
    
    def __init__(self):
        super().__init__()

    def learn(self, data, **kwargs):
        """
        Set up and run the CORL1 algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        """
        config, _ = get_config()
        for k in kwargs:
            config.__dict__[k] = kwargs[k]

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

        logger.info('Python version is {}'.format(platform.python_version()))

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
        logger.info('Finished creating training dataset, actor model and reward class')

        logger.info('Starting session...')
        sess_config = tf.ConfigProto(log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())

            logger.info('Shape of actor.input: {}'.format(sess.run(tf.shape(actor.input_))))

            reward_max_per_batch = []
            reward_mean_per_batch = []
            max_rewards = []
            max_reward = float('-inf')

            max_sum = 0

            logger.info('Starting training.')
            loss1_s, loss_2s = [], []
            
            for i in (range(1, config.nb_epoch+1)):
                
                # if i%1000 == 0 or i%1000 == 1:
                #     logger.info('time log_3')

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
                            logger.info('[iter {}] [Batch {}_th] The optimal nodes order cover true graph {}/{}!'.format(i,m,max_sum,actor.max_length*actor.max_length))
                
                # if i % 1000 == 0:
                #     logger.info('time log_1')

                graphs_feed = np.stack(samples)
                action_mask_s = np.stack(action_mask_s)
                reward_feed = callreward.cal_rewards(graphs_feed, positions)
                
                # if i % 1000 == 0:
                #     logger.info('time log_2')

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
                    logger.info('[iter {}], max_reward: {:.4}, max_reward_batch: {:.4}'.format(i, max_reward, max_reward_batch))

                if i == 1 or (i+1) % config.lambda_iter_num == 0:
                    ls_kv = callreward.update_all_scores()

                    score_min, graph_int_key = ls_kv[0][1][0], ls_kv[0][0]
                    logger.info('[iter {}] score_min {:.4}'.format(i, score_min*1.0))
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
                        
                        logger.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                        logger.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))

            plt.figure(1)
            plt.plot(reward_max_per_batch, label='max reward per batch')
            plt.plot(reward_mean_per_batch, label='mean reward per batch')
            plt.plot(max_rewards, label='max reward')
            plt.legend()
            plt.savefig('reward_batch_average.png')
            plt.show()
            plt.close()

            logger.info('Training COMPLETED!')
        return graph_batch_pruned.T
