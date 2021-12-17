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


import torch
import torch.nn.functional as F

from ..utils.validation import Validation


def episodic_actor_loss(td_target, prediction_env, log_softmax,
                        device=None) -> torch.Tensor:
    """Calculate actor loss for reward type is episodic"""

    td_target, prediction_env, log_softmax = Validation.to_device(td_target,
                                                            prediction_env,
                                                            log_softmax,
                                                            device=device)
    prediction_env_no_grad = prediction_env.detach()
    advantage_no_grad = td_target - prediction_env_no_grad.T
    step_loss = advantage_no_grad * log_softmax[:-1]
    actor_loss = - torch.mean(step_loss)

    return actor_loss


def episodic_critic_loss(td_target, prediction_env,
                         device=None) -> torch.Tensor:
    """Calculate critic loss for reward type is 'episodic'"""

    td_target, prediction_env = Validation.to_device(td_target, prediction_env,
                                                     device=device)
    advantage = td_target - prediction_env.T
    td_error = advantage.reshape((-1, 1)).squeeze()
    critic_loss = torch.mean(torch.square(td_error))

    return critic_loss


def dense_actor_loss(reward, avg_baseline, predict_reward, log_softmax,
                     device=None) -> torch.Tensor:
    """Calculate actor loss for reward type is 'dense'"""

    reward, avg_baseline, predict_reward, log_softmax = Validation.to_device(
        reward, avg_baseline, predict_reward, log_softmax, device=device
    )
    predict_reward = predict_reward.detach() # [Batch size, 1]
    reward_baseline = reward - avg_baseline - predict_reward
    actor_loss = - torch.mean(reward_baseline * log_softmax, 0)

    return actor_loss


def dense_critic_loss(reward, avg_baseline, predict_reward,
                      device=None) -> torch.Tensor:
    """Calculate actor loss for reward type is 'dense'"""

    reward, avg_baseline, predict_reward = Validation.to_device(
        reward, avg_baseline, predict_reward, device=device
    )
    reward = reward.detach()
    critic_loss = F.mse_loss(reward - avg_baseline,  predict_reward)

    return critic_loss

