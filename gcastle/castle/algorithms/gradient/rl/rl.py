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

import logging
import platform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from .data_loader import DataGenerator_read_data
from .models import Actor
from .rewards import get_Reward
from .helpers.tf_utils import set_seed
from .helpers.config_graph import get_config
from .helpers.lambda_utils import BIC_lambdas
from .helpers.analyze_utils import convert_graph_int_to_adj_mat, \
    graph_prunned_by_coef, graph_prunned_by_coef_2nd

from castle.common import BaseLearner, Tensor
from castle.metrics import MetricsDAG


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class RL(BaseLearner):
    """
    RL Algorithm.
    A classic causal discovery algorithm based on conditional independence tests.

    Attributes
    ----------
    causal_matrix : numpy.ndarray
        Learned causal structure matrix

    References
    ----------
    https://arxiv.org/abs/1906.04477

    Examples
    --------
    >>> from castle.algorithms import RL
    >>> from castle.datasets import load_dataset
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> true_dag, X = load_dataset(name='iid_test')
    >>> n = RL(lambda_flag_default=True)
    >>> n.learn(X, dag=true_dag)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self, **kwargs):
        super().__init__()

        config, _ = get_config()
        for k in kwargs:
            config.__dict__[k] = kwargs[k]
        
        self.config = config

    def learn(self, data, dag=None):
        """
        Set up and run the RL algorithm.

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
        # Reproducibility
        set_seed(config.seed)

        logging.info('Python version is {}'.format(platform.python_version()))

        # input data
        if hasattr(config, 'dag'):
            training_set = DataGenerator_read_data(
                X, config.dag, config.normalize, config.transpose)
        else:
            training_set = DataGenerator_read_data(
                X, None, config.normalize, config.transpose)

        # set penalty weights
        score_type = config.score_type
        reg_type = config.reg_type

        if config.lambda_flag_default:            
            sl, su, strue = BIC_lambdas(training_set.inputdata, None, None, None, reg_type, score_type)
            lambda1 = 0
            lambda1_upper = 5
            lambda1_update_add = 1
            lambda2 = 1/(10**(np.round(config.max_length/3)))
            lambda2_upper = 0.01
            lambda2_update_mul = 10
            lambda_iter_num = config.lambda_iter_num

            # test initialized score
            logging.info('Original sl: {}, su: {}, strue: {}'.format(sl, su, strue))
            logging.info('Transfomed sl: {}, su: {}, lambda2: {}, true: {}'.format(sl, su, lambda2,
                        (strue-sl)/(su-sl)*lambda1_upper))   
        else:
            # test choices for the case with mannualy provided bounds
            # not fully tested
            sl = config.score_lower
            su = config.score_upper
            if config.score_bd_tight:
                lambda1 = 2
                lambda1_upper = 2
            else:
                lambda1 = 0
                lambda1_upper = 5
                lambda1_update_add = 1
            lambda2 = 1/(10**(np.round(config.max_length/3)))
            lambda2_upper = 0.01
            lambda2_update_mul = config.lambda2_update
            lambda_iter_num = config.lambda_iter_num

        # actor
        actor = Actor(config)
        callreward = get_Reward(actor.batch_size, config.max_length, 
                                actor.input_dimension, training_set.inputdata,
                                sl, su, lambda1_upper, score_type, reg_type, 
                                config.l1_graph_reg, False)
        logging.info('Finished creating training dataset, actor model and reward class')

        logging.info('Starting session...')
        sess_config = tf.ConfigProto(log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            # Run initialize op
            sess.run(tf.global_variables_initializer())

            # Test tensor shape
            logging.info('Shape of actor.input: {}'.format(sess.run(tf.shape(actor.input_))))

            # Initialize useful variables
            rewards_avg_baseline = []
            rewards_batches = []
            reward_max_per_batch = []
            
            lambda1s = []
            lambda2s = []
            
            graphss = []
            probsss = []
            max_rewards = []
            max_reward = float('-inf')
            max_reward_score_cyc = (lambda1_upper+1, 0)

            logging.info('Starting training.')
            
            for i in (range(1, config.nb_epoch + 1)):

                if config.verbose:
                    logging.info('Start training for {}-th epoch'.format(i))

                input_batch = training_set.train_batch(actor.batch_size, actor.max_length, actor.input_dimension)
                graphs_feed = sess.run(actor.graphs, feed_dict={actor.input_: input_batch})
                reward_feed = callreward.cal_rewards(graphs_feed, lambda1, lambda2)

                # max reward, max reward per batch
                max_reward = -callreward.update_scores([max_reward_score_cyc], lambda1, lambda2)[0]
                max_reward_batch = float('inf')
                max_reward_batch_score_cyc = (0, 0)

                for reward_, score_, cyc_ in reward_feed:
                    if reward_ < max_reward_batch:
                        max_reward_batch = reward_
                        max_reward_batch_score_cyc = (score_, cyc_)
                            
                max_reward_batch = -max_reward_batch

                if max_reward < max_reward_batch:
                    max_reward = max_reward_batch
                    max_reward_score_cyc = max_reward_batch_score_cyc

                # for average reward per batch
                reward_batch_score_cyc = np.mean(reward_feed[:,1:], axis=0)

                if config.verbose:
                    logging.info('Finish calculating reward for current batch of graph')

                # Get feed dict
                feed = {actor.input_: input_batch, actor.reward_: -reward_feed[:,0], actor.graphs_:graphs_feed}

                summary, base_op, score_test, probs, graph_batch, reward_batch, \
                    reward_avg_baseline, train_step1, train_step2 = sess.run( \
                        [actor.merged, actor.base_op, actor.test_scores, \
                         actor.log_softmax, actor.graph_batch, actor.reward_batch, \
                         actor.avg_baseline, actor.train_step1, actor.train_step2], \
                        feed_dict=feed)

                if config.verbose:
                    logging.info('Finish updating actor and critic network using reward calculated')
                
                lambda1s.append(lambda1)
                lambda2s.append(lambda2)

                rewards_avg_baseline.append(reward_avg_baseline)
                rewards_batches.append(reward_batch_score_cyc)
                reward_max_per_batch.append(max_reward_batch_score_cyc)

                graphss.append(graph_batch)
                probsss.append(probs)
                max_rewards.append(max_reward_score_cyc)

                # logging
                if i == 1 or i % 500 == 0:
                    logging.info('[iter {}] reward_batch: {:.4}, max_reward: {:.4}, max_reward_batch: {:.4}'.format(i,
                                reward_batch, max_reward, max_reward_batch))

                # update lambda1, lamda2
                if i == 1 or i % lambda_iter_num == 0:
                    ls_kv = callreward.update_all_scores(lambda1, lambda2)

                    graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

                    if cyc_min < 1e-5:
                        lambda1_upper = score_min
                    lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
                    lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
                    logging.info('[iter {}] lambda1 {:.4}, upper {:.4}, lambda2 {:.4}, upper {:.4}, score_min {:.4}, cyc_min {:.4}'.format(i,
                                lambda1*1.0, lambda1_upper*1.0, lambda2*1.0, lambda2_upper*1.0, score_min*1.0, cyc_min*1.0))

                    graph_batch = convert_graph_int_to_adj_mat(graph_int)

                    if reg_type == 'LR':
                        graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, training_set.inputdata))
                    elif reg_type == 'QR':
                        graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))
                    # elif reg_type == 'GPR':
                    #     # The R codes of CAM pruning operates the graph form that (i,j)=1 indicates i-th node-> j-th node
                    #     # so we need to do a tranpose on the input graph and another tranpose on the output graph
                    #     graph_batch_pruned = np.transpose(pruning_cam(training_set.inputdata, np.array(graph_batch).T))

                    # estimate accuracy
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
            plt.plot(rewards_batches, label='reward per batch')
            plt.plot(max_rewards, label='max reward')
            plt.legend()
            plt.savefig('reward_batch_average.png')
            plt.show()
            plt.close()
            
            logging.info('Training COMPLETED !')

        return graph_batch_pruned.T
