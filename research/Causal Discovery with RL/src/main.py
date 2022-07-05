#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python codes for 'Causal Discovery with Reinforcement Learning', ICLR 2020 (oral)
Authors: Shengyu Zhu, Huawei Noah's Ark Lab,
         Ignavier Ng, University of Toronto (work was done during an internship at Huawei Noah's Ark Lab)
         Zhitang Chen, Huawei Noah's Ark Lab
"""

import os
import logging
import platform
import random
import numpy as np
import pandas as pd
from pytz import timezone
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import DataGenerator_read_data
from models import Actor
from rewards import get_Reward
from helpers.config_graph import get_config, print_config
from helpers.dir_utils import create_dir
from helpers.log_helper import LogHelper
from helpers.tf_utils import set_seed
from helpers.analyze_utils import convert_graph_int_to_adj_mat, graph_prunned_by_coef, \
                                  count_accuracy, graph_prunned_by_coef_2nd
from helpers.cam_with_pruning_cam import pruning_cam
from helpers.lambda_utils import BIC_lambdas

# Configure matplotlib for plotting
import matplotlib
matplotlib.use('Agg')


def main():
    # Setup for output directory and logging
    output_dir = 'output/{}'.format(datetime.now(timezone('Asia/Hong_Kong')).strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3])
    create_dir(output_dir)
    LogHelper.setup(log_path='{}/training.log'.format(output_dir),
                    level_str='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info('Python version is {}'.format(platform.python_version()))
    _logger.info('Current commit of code: ___')

    # Get running configuration
    config, _ = get_config()
    config.save_model_path = '{}/model'.format(output_dir)
    # config.restore_model_path = '{}/model'.format(output_dir)
    config.summary_dir = '{}/summary'.format(output_dir)
    config.plot_dir = '{}/plot'.format(output_dir)
    config.graph_dir = '{}/graph'.format(output_dir)

    # Create directory
    create_dir(config.summary_dir)
    create_dir(config.summary_dir)
    create_dir(config.plot_dir)
    create_dir(config.graph_dir)

    # Reproducibility
    set_seed(config.seed)

    # Log the configuration parameters
    _logger.info('Configuration parameters: {}'.format(vars(config)))    # Use vars to convert config to dict for logging
    
    if config.read_data:
        file_path = '{}/data.npy'.format(config.data_path)
        solution_path = '{}/DAG.npy'.format(config.data_path)
        training_set = DataGenerator_read_data(file_path, solution_path, config.normalize, config.transpose)
    else:
        raise ValueError("Only support importing data from existing files")
        
    # set penalty weights
    score_type = config.score_type
    reg_type = config.reg_type
    
    if config.lambda_flag_default:
        
        sl, su, strue = BIC_lambdas(training_set.inputdata, None, None, training_set.true_graph.T, reg_type, score_type)
        
        lambda1 = 0
        lambda1_upper = 5
        lambda1_update_add = 1
        lambda2 = 1/(10**(np.round(config.max_length/3)))
        lambda2_upper = 0.01
        lambda2_update_mul = 10
        lambda_iter_num = config.lambda_iter_num

        # test initialized score
        _logger.info('Original sl: {}, su: {}, strue: {}'.format(sl, su, strue))
        _logger.info('Transfomed sl: {}, su: {}, lambda2: {}, true: {}'.format(sl, su, lambda2,
                     (strue-sl)/(su-sl)*lambda1_upper))
        
    else:
        # test choices for the case with manually provided bounds
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

    callreward = get_Reward(actor.batch_size, config.max_length, actor.input_dimension, training_set.inputdata,
                            sl, su, lambda1_upper, score_type, reg_type, config.l1_graph_reg, False)

    _logger.info('Finished creating training dataset, actor model and reward class')

    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)  

    _logger.info('Starting session...')
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Test tensor shape
        _logger.info('Shape of actor.input: {}'.format(sess.run(tf.shape(actor.input_))))
        # _logger.info('training_set.true_graph: {}'.format(training_set.true_graph))
        # _logger.info('training_set.b: {}'.format(training_set.b))

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
        image_count = 0
        
        accuracy_res = []
        accuracy_res_pruned = []
        
        max_reward_score_cyc = (lambda1_upper+1, 0)

        # Summary writer
        writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

        _logger.info('Starting training.')
            
        for i in (range(1, config.nb_epoch + 1)):

            if config.verbose:
                _logger.info('Start training for {}-th epoch'.format(i))

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
                _logger.info('Finish calculating reward for current batch of graph')

            # Get feed dict
            feed = {actor.input_: input_batch, actor.reward_: -reward_feed[:,0], actor.graphs_:graphs_feed}

            summary, base_op, score_test, probs, graph_batch, \
                reward_batch, reward_avg_baseline, train_step1, train_step2 = sess.run([actor.merged, actor.base_op,
                actor.test_scores, actor.log_softmax, actor.graph_batch, actor.reward_batch, actor.avg_baseline, actor.train_step1,
                actor.train_step2], feed_dict=feed)

            if config.verbose:
                _logger.info('Finish updating actor and critic network using reward calculated')
                    
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
                if i >= 500:
                    writer.add_summary(summary,i)
                    
                _logger.info('[iter {}] reward_batch: {}, max_reward: {}, max_reward_batch: {}'.format(i,
                             reward_batch, max_reward, max_reward_batch))
                # other logger info; uncomment if you want to check
                # _logger.info('graph_batch_avg: {}'.format(graph_batch))
                # _logger.info('graph true: {}'.format(training_set.true_graph))
                # _logger.info('graph weights true: {}'.format(training_set.b))
                # _logger.info('=====================================')

                plt.figure(1)
                plt.plot(rewards_batches, label='reward per batch')
                plt.plot(max_rewards, label='max reward')
                plt.legend()
                plt.savefig('{}/reward_batch_average.png'.format(config.plot_dir))
                plt.close()

                image_count += 1
                # this draw the average graph per batch. 
                # can be modified to draw the graph (with or w/o pruning) that has the best reward
                fig = plt.figure(2)
                fig.suptitle('Iteration: {}'.format(i))
                ax = fig.add_subplot(1, 2, 1)
                ax.set_title('recovered_graph')
                ax.imshow(np.around(graph_batch.T).astype(int),cmap=plt.cm.gray)
                ax = fig.add_subplot(1, 2, 2)
                ax.set_title('ground truth')
                ax.imshow(training_set.true_graph, cmap=plt.cm.gray)
                plt.savefig('{}/recovered_graph_iteration_{}.png'.format(config.plot_dir, image_count))
                plt.close()
                    
            # update lambda1, lamda2
            if (i+1) % lambda_iter_num == 0:
                ls_kv = callreward.update_all_scores(lambda1, lambda2)
                # np.save('{}/solvd_dict_epoch_{}.npy'.format(config.graph_dir, i), np.array(ls_kv))
                max_rewards_re = callreward.update_scores(max_rewards, lambda1, lambda2)
                rewards_batches_re = callreward.update_scores(rewards_batches, lambda1, lambda2)
                reward_max_per_batch_re = callreward.update_scores(reward_max_per_batch, lambda1, lambda2)

                # saved somewhat more detailed logging info
                np.save('{}/solvd_dict.npy'.format(config.graph_dir), np.array(ls_kv))
                pd.DataFrame(np.array(max_rewards_re)).to_csv('{}/max_rewards.csv'.format(output_dir))
                pd.DataFrame(rewards_batches_re).to_csv('{}/rewards_batch.csv'.format(output_dir))
                pd.DataFrame(reward_max_per_batch_re).to_csv('{}/reward_max_batch.csv'.format(output_dir))
                pd.DataFrame(lambda1s).to_csv('{}/lambda1s.csv'.format(output_dir))
                pd.DataFrame(lambda2s).to_csv('{}/lambda2s.csv'.format(output_dir))
                    
                graph_int, score_min, cyc_min = np.int32(ls_kv[0][0]), ls_kv[0][1][1], ls_kv[0][1][-1]

                if cyc_min < 1e-5:
                    lambda1_upper = score_min
                lambda1 = min(lambda1+lambda1_update_add, lambda1_upper)
                lambda2 = min(lambda2*lambda2_update_mul, lambda2_upper)
                _logger.info('[iter {}] lambda1 {}, upper {}, lambda2 {}, upper {}, score_min {}, cyc_min {}'.format(i+1,
                             lambda1, lambda1_upper, lambda2, lambda2_upper, score_min, cyc_min))
                    
                graph_batch = convert_graph_int_to_adj_mat(graph_int)

                if reg_type == 'LR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef(graph_batch, training_set.inputdata))
                elif reg_type == 'QR':
                    graph_batch_pruned = np.array(graph_prunned_by_coef_2nd(graph_batch, training_set.inputdata))
                elif reg_type == 'GPR':
                    # The R codes of CAM pruning operates the graph form that (i,j)=1 indicates i-th node-> j-th node
                    # so we need to do a tranpose on the input graph and another tranpose on the output graph
                    graph_batch_pruned = np.transpose(pruning_cam(training_set.inputdata, np.array(graph_batch).T))

                # estimate accuracy
                acc_est = count_accuracy(training_set.true_graph, graph_batch.T)
                acc_est2 = count_accuracy(training_set.true_graph, graph_batch_pruned.T)

                fdr, tpr, fpr, shd, nnz = acc_est['fdr'], acc_est['tpr'], acc_est['fpr'], acc_est['shd'], \
                                          acc_est['pred_size']
                fdr2, tpr2, fpr2, shd2, nnz2 = acc_est2['fdr'], acc_est2['tpr'], acc_est2['fpr'], acc_est2['shd'], \
                                               acc_est2['pred_size']
                    
                accuracy_res.append((fdr, tpr, fpr, shd, nnz))
                accuracy_res_pruned.append((fdr2, tpr2, fpr2, shd2, nnz2))
                
                np.save('{}/accuracy_res.npy'.format(output_dir), np.array(accuracy_res))
                np.save('{}/accuracy_res2.npy'.format(output_dir), np.array(accuracy_res_pruned))
                    
                _logger.info('before pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr, tpr, fpr, shd, nnz))
                _logger.info('after  pruning: fdr {}, tpr {}, fpr {}, shd {}, nnz {}'.format(fdr2, tpr2, fpr2, shd2, nnz2))

            # Save the variables to disk
            if i % max(1, int(config.nb_epoch / 5)) == 0 and i != 0:
                curr_model_path = saver.save(sess, '{}/tmp.ckpt'.format(config.save_model_path), global_step=i)
                _logger.info('Model saved in file: {}'.format(curr_model_path))

        _logger.info('Training COMPLETED !')
        saver.save(sess, '{}/actor.ckpt'.format(config.save_model_path))


if __name__ == '__main__':
    main()
