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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class BaseEncoder(nn.Module):
    """Base class for network"""

    def __init__(self, input_dim, embed_dim, hidden_dim=1024,
                 device=None) -> None :
        super(BaseEncoder, self).__init__()
        if embed_dim is None:
            embed_dim = input_dim

        # this layer just for Encoder
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,
                      out_channels=embed_dim,
                      kernel_size=(1,),
                      device=device),
        )
        if self.__class__.__name__ in ['TransformerEncoder', 'MLPEncoder']:
            # this layer just for ``TransformerEncoder`` and ``MLPEncoder``.
            self.feedforward_conv1d = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim,
                          out_channels=hidden_dim,
                          kernel_size=(1, ),
                          device=device),
                nn.ReLU(),
                nn.Conv1d(in_channels=hidden_dim,
                          out_channels=embed_dim,
                          kernel_size=(1,),
                          device=device),
            )
        self.bn = nn.BatchNorm1d(num_features=embed_dim, device=device)


class BaseDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, device=None) -> None:
        # input_dim is embed_dim
        super(BaseDecoder, self).__init__()
        if self.__class__.__name__ == 'MLPDecoder':
            # this layer just for MLPDecoder
            self.feedforward_mlp = nn.Sequential(
                nn.Linear(in_features=input_dim,
                          out_features=hidden_dim,
                          device=device),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim,
                          out_features=input_dim,
                          device=device)
            )
        self.bn = nn.BatchNorm1d(num_features=input_dim, device=device)


class PointerDecoder(BaseDecoder):
    """Base Class for all Decoder"""

    def __init__(self, input_dim, hidden_dim, device=None) -> None:
        super(PointerDecoder, self).__init__(input_dim=input_dim,
                                             hidden_dim=hidden_dim,
                                             device=device)
        self.hidden_dim = hidden_dim
        self.device = device
        self.positions = []  # store visited cities for reward
        self.mask = torch.zeros(1, device=self.device)
        self.mask_scores = []
        # Attention mechanism -- glimpse  _encoder_glimpse
        self.conv1d_ref_g = nn.Conv1d(in_channels=input_dim,
                                 out_channels=hidden_dim,
                                 kernel_size=(1,),
                                 device=device)
        self.w_q_g = nn.Linear(in_features=hidden_dim,
                                out_features=hidden_dim,
                                bias=False,
                                device=device)
        self.v_g = nn.Linear(in_features=hidden_dim,
                             out_features=1,
                             bias=False,
                             device=device)
        # Pointer mechanism  _encoder_pointer
        self.conv1d_ref_p = nn.Conv1d(in_channels=input_dim,
                                  out_channels=hidden_dim,
                                  kernel_size=(1,),
                                  device=device)
        self.w_q_p = nn.Linear(in_features=hidden_dim,
                             out_features=hidden_dim,
                              bias=False,
                              device=device)
        self.v_p = nn.Linear(in_features=hidden_dim,
                           out_features=1,
                           bias=False,
                           device=device)

    def step_decode(self, input, state=None) -> tuple:
        """

        Parameters
        ----------
        input:
            Encoder's output
        state: tuple, None
            (h, c) for Pointer and None for MLP

        Returns
        -------
        output: tuple

        """

        if self.__class__.__name__ == 'LSTMDecoder':
            # Run the cell on a combination of the previous input and state
            h, c = self.lstm_cell(input, state)
            output = h
            state = (h, c)
        elif self.__class__.__name__ == 'MLPDecoder':
            output = self.mlp(input).squeeze()
        else:
            raise TypeError(f'Supported subclass of PointerDecoder is one of '
                            f'[`LSTMDecoder`, `MLPDecoder`], but got'
                            f'``{self.__class__.__name__}``.')

        # [batch_size, time_sequence]
        masked_scores = self.pointer_net(self.encoder_output, output)

        # Multinomial distribution
        prob = Categorical(logits=masked_scores)

        # Sample from distribution
        action = prob.sample().long()

        self.mask = self.mask + F.one_hot(action, self.seq_length)

        # Retrieve decoder's new input
        action_index = action.reshape(-1, 1, 1).repeat(1, 1, self.hidden_dim)
        next_input = torch.gather(self.encoder_output, 0, action_index)[:, 0, :]

        return next_input, state, action, masked_scores

    def pointer_net(self, ref, query) -> torch.Tensor:
        """Attention mechanism + Pointer mechanism

        Parameters
        ----------
        ref: torch.Tensor
            encoder_states
        query: torch.Tensor
            decoder_states
        """

        # Attention mechanism
        encoder_ref_g = self.conv1d_ref_g(
            ref.permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoder_query_g = self.w_q_g(query).unsqueeze(1)
        scores_g = torch.mean(
            self.v_g(torch.tanh(encoder_ref_g + encoder_query_g)), dim=-1
        )
        attention_g = F.softmax(scores_g - self.mask * 1e9, dim=-1)

        glimpse = torch.mul(ref, attention_g.unsqueeze(2))
        glimpse = torch.sum(glimpse, dim=1) + query

        # Pointer mechanism
        encoder_ref_p = self.conv1d_ref_p(
            ref.permute(0, 2, 1)
        ).permute(0, 2, 1)
        encoder_query_p = self.w_q_p(glimpse).unsqueeze(1)
        scores_p = torch.mean(
            self.v_p(torch.tanh(encoder_ref_p + encoder_query_p)), dim=-1
        )
        if self.__class__.__name__ == 'MLPDecoder':
            scores_p = 10.0 * torch.tanh(scores_p)
        masked_scores = scores_p - self.mask * 1e9

        return masked_scores

    def log_softmax(self, input, position, mask, state_0, state_1) -> torch.Tensor:

        if self.__class__.__name__ == 'LSTMDecoder':
            state = state_0, state_1
            h, c = self.lstm_cell(input, state)
            output = h
        elif self.__class__.__name__ == 'MLPDecoder':
            output = self.mlp(input)
        else:
            raise TypeError(f'Supported subclass of PointerDecoder is one of '
                            f'[`LSTMDecoder`, `MLPDecoder`], but got'
                            f'``{self.__class__.__name__}``.')
        self.mask = torch.tensor(mask, dtype=torch.float32, device=self.device)

        # encoder_output_expand
        encoder_output_ex = self.encoder_output.unsqueeze(1)
        encoder_output_ex = encoder_output_ex.repeat(1, self.seq_length, 1, 1)
        encoder_output_ex = encoder_output_ex.reshape(-1, self.seq_length, self.hidden_dim)
        masked_scores = self.pointer_net(encoder_output_ex, output)

        prob = Categorical(logits=masked_scores)
        log_softmax = prob.log_prob(position.reshape(-1,))
        self.mask = torch.zeros(1, device=self.device)

        return log_softmax
