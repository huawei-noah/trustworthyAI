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
from torch import Tensor

from ..models.encoders import LSTMEncoder, TransformerEncoder, MLPEncoder
from ..models.decoders import LSTMDecoder, MLPDecoder


class Actor(object):
    """
    Design of Actor Part in Reinforcement Learning Actor-Critic Algorithm.

    Include ``Encoder`` and ``Decoder``. The ``Encoder`` is used to map the
    observed data to the embedding space S={s1, · · · , sd}.
    The ``Decoder`` maps the state space S^(S_hat) to the action space A.

    Parameters
    ----------
    input_dim: int
        dimension of input data, number of variables, number of DAG node.
    embed_dim: int, default: 256
        dimension of embedding space S.
    encoder_blocks: int, default: 3
        Effective when `encoder`='transformer'.
        Design for the neural network structure of the Transformer encoder,
        each block is composed of a multi-head attention network and
        feed-forward neural networks.
    encoder_heads: int, default: 8
        Effective when `encoder_name`='transformer'.
        head number of multi-head attention network,
    encoder_name: str, default: 'transformer'
        Indicates type of encoder, one of [`transformer`, `lstm`, `mlp`]
    decoder_name: str, default: 'lstm'
        Indicates type of decoder, one of [`lstm`, `mlp`]
    """

    ENCODER_HIDDEN_DIM = 1024

    def __init__(self, input_dim, embed_dim=256,
                 encoder_name='transformer',
                 encoder_blocks=3,
                 encoder_heads=8,
                 decoder_name='lstm',
                 device=None) -> None:

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.encoder_blocks = encoder_blocks
        self.encoder_heads = encoder_heads
        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.device = device

        self._instantiation()

    def _instantiation(self):
        if self.encoder_name.lower() == 'transformer':
            self.encoder = TransformerEncoder(input_dim=self.input_dim,
                                              embed_dim=self.embed_dim,
                                              hidden_dim=self.ENCODER_HIDDEN_DIM,
                                              heads=self.encoder_heads,
                                              blocks=self.encoder_blocks,
                                              device=self.device)
        elif self.encoder_name.lower() == 'lstm':
            self.encoder = LSTMEncoder(input_dim=self.input_dim,
                                embed_dim=self.embed_dim,
                                device=self.device)
        elif self.encoder_name.lower() == 'mlp':
            self.encoder = MLPEncoder(input_dim=self.input_dim,
                              embed_dim=self.embed_dim,
                              hidden_dim=self.ENCODER_HIDDEN_DIM,
                              device=self.device)
        else:
            raise ValueError(f'Invalid encoder type, expected one of '
                             f'[`transformer`, `lstm`, `mlp`], but got'
                             f'``{self.encoder_name}``.')

        if self.decoder_name.lower() == 'lstm':
            self.decoder = LSTMDecoder(input_dim=self.embed_dim,
                                hidden_dim=self.embed_dim,
                                device=self.device)
        elif self.decoder_name.lower() == 'mlp':
            self.decoder = MLPDecoder(input_dim=self.embed_dim,
                              hidden_dim=self.embed_dim,
                              device=self.device)
        else:
            raise ValueError(f'Invalid decoder type, expected one of '
                             f'[`lstm`, `mlp`], but got ``{self.decoder_name}``.')

    def encode(self, input) -> torch.Tensor:
        """
        draw a batch of samples from X, encode them to S and calculate
        the initial state ˆs0

        Parameters
        ----------
        input: Tensor
            a batch samples from X

        Returns
        -------
        out: Tensor
            encoder_output.shape=(batch_size, n_nodes, embed_dim)
        """

        self.encoder_output = self.encoder(input)

        return self.encoder_output

    def decode(self, input) -> torch.Tensor:
        """
        Maps the state space ˆS to the action space A.

        Parameters
        ----------
        input: Tensor
            (batch_size, n_nodes, input_dim)
            a batch of samples from X, output of Encoder.

        Returns
        -------
        out: tuple
            (actions, mask_scores, s_list, h_list, c_list)

            Notes::
                actions: (batch_size, n_nodes)
                mask_scores: (batch_size, n_nodes, n_nodes)
                s_list: input for lstm cell, (batch_size, n_nodes, embed_dim)
                h_list: h for lstm cell, (batch_size, n_nodes, embed_dim)
                c_list: c for lstm cell, (batch_size, n_nodes, embed_dim)
        """

        outputs = self.decoder(input)

        return outputs
