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
import argparse
import platform
import torch
import numpy as np

from .data_loader import DataGenerator_read_data
from .models import Actor
from .rewards import get_Reward
from .helpers.torch_utils import set_seed, is_cuda_available
from .helpers.lambda_utils import BIC_lambdas
from .helpers.analyze_utils import convert_graph_int_to_adj_mat, \
    graph_prunned_by_coef, graph_prunned_by_coef_2nd

from castle.common import BaseLearner, Tensor
from castle.metrics import MetricsDAG


class RL(BaseLearner):
    """
    RL Algorithm.
    A RL-based algorithm that can work with flexible score functions (including non-smooth ones).

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
        activation for decoder
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
    inference_mode: bool
        switch to inference mode when model is trained
    verbose: bool
        print detailed logging or not
    device_type: str
        whether to use GPU or not
    device_ids: int
        choose which gpu to use

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
    >>> X, true_dag, _ = load_dataset('IID_Test')
    >>> n = RL()
    >>> n.learn(X, dag=true_dag)
    >>> GraphDAG(n.causal_matrix, true_dag)
    >>> met = MetricsDAG(n.causal_matrix, true_dag)
    >>> print(met.metrics)
    """
    
    def __init__(self, encoder_type='TransformerEncoder', 
                 hidden_dim=64, 
                 num_heads=16, 
                 num_stacks=6, 
                 residual=False, 
                 decoder_type='SingleLayerDecoder', 
                 decoder_activation='tanh', 
                 decoder_hidden_dim=16, 
                 use_bias=False, 
                 use_bias_constant=False, 
                 bias_initial_value=False, 
                 batch_size=64, 
                 input_dimension=64, 
                 normalize=False, 
                 transpose=False, 
                 score_type='BIC', 
                 reg_type='LR', 
                 lambda_iter_num=1000, 
                 lambda_flag_default=True, 
                 score_bd_tight=False, 
                 lambda1_update=1.0, 
                 lambda2_update=10, 
                 score_lower=0.0, 
                 score_upper=0.0, 
                 lambda2_lower=-1.0, 
                 lambda2_upper=-1.0, 
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
                 inference_mode=True, 
                 verbose=False, 
                 device_type='cpu', 
                 device_ids=0):

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
        self.config.inference_mode = inference_mode
        self.config.verbose = verbose
        self.config.device_type = device_type
        self.config.device_ids = device_ids

        if not is_cuda_available:
            self.config.device_type = 'cpu'

        if self.config.device_type == 'gpu':
            self.config.device = torch.device(type="cuda", index=self.config.device_ids)

    def learn(self, data, columns=None, dag=None, **kwargs):
        """
        Set up and run the RL algorithm.

        Parameters
        ----------
        data: castle.Tensor or numpy.ndarray
            The castle.Tensor or numpy.ndarray format data you want to learn.
        columns : Index or array-like
            Column labels to use for resulting tensor. Will default to
            RangeIndex (0, 1, 2, ..., n) if no column labels are provided.
        dag : ndarray
            two-dimensional, prior matrix
        """
        config = self.config
        if dag is not None:
            config.dag = dag

        X = Tensor(data, columns=columns)
        
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
        callreward = get_Reward(config.batch_size, config.max_length, 
                                config.input_dimension, training_set.inputdata,
                                sl, su, lambda1_upper, score_type, reg_type, 
                                config.l1_graph_reg, False)
        logging.info('Finished creating training dataset and reward class')

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

            input_batch = training_set.train_batch(config.batch_size, config.max_length, config.input_dimension)
            inputs = torch.from_numpy(np.array(input_batch))
            if config.device_type == 'gpu':
                inputs = inputs.cuda(config.device_ids)

            # Test tensor shape
            if i == 1:
                logging.info('Shape of actor.input: {}'.format(inputs.shape))

            # actor
            actor.build_permutation(inputs)
            graphs_feed = actor.graphs_
            if config.device_type == 'gpu':
                reward_feed = callreward.cal_rewards(graphs_feed.cpu().detach().numpy(), lambda1, lambda2)  # np.array
                actor.build_reward(reward_ = -torch.from_numpy(reward_feed)[:,0].cuda(config.device_ids))
            else:
                reward_feed = callreward.cal_rewards(graphs_feed.detach().numpy(), lambda1, lambda2)  # np.array
                actor.build_reward(reward_ = -torch.from_numpy(reward_feed)[:,0])

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

            score_test, probs, graph_batch, \
            reward_batch, reward_avg_baseline = \
                    actor.test_scores, actor.log_softmax, actor.graph_batch, \
                    actor.reward_batch, actor.avg_baseline

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

                graph_int, score_min, cyc_min = np.int64(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

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
        
        logging.info('Training COMPLETED !')

        return graph_batch_pruned.T
