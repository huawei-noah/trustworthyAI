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
import torch.distributions as distr


class BilinearDecoder(nn.Module):

    def __init__(self, batch_size, max_length, hidden_dim, use_bias,
                 bias_initial_value, use_bias_constant, is_train,
                 device=None):

        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = hidden_dim
        self.input_embed = hidden_dim    # dimension of embedding space (actor)
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant
        self.device = device
        self.is_training = is_train

        self._W = nn.Parameter(torch.Tensor(*(self.input_embed, self.input_embed)).to(self.device))
        self._l = nn.Parameter(torch.Tensor(1).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._W)  # variables initializer

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = torch.randn([1]).numpy()[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        nn.init.constant_(self._l, bias_initial_value)

    def forward(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]

        W = self._W

        logits = torch.einsum('ijk, kn, imn->ijm', encoder_output, W, encoder_output)    # Readability

        self.logit_bias =  self._l

        if self.use_bias:    # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

        for i in range(self.max_length):

            position = torch.ones([encoder_output.shape[0]],
                                  device=self.device) * i
            position = position.long()

            # Update mask
            self.mask = torch.zeros((encoder_output.shape[0], self.max_length),
                                    device=self.device).scatter_(1, position.view(encoder_output.shape[0], 1), 1)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
            sampled_arr.requires_grad=True

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy


class NTNDecoder(nn.Module):

    def __init__(self, batch_size, max_length, hidden_dim,
                 decoder_hidden_dim, decoder_activation, use_bias,
                 bias_initial_value, use_bias_constant, is_train, device=None):
        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = hidden_dim
        self.input_embed = hidden_dim  # dimension of embedding space (actor)
        self.max_length = max_length
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_activation = decoder_activation
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant
        self.is_training = is_train
        self.device = device

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            self.activation = nn.Tanh()
        elif self.decoder_activation == 'relu':
            self.activation = nn.ReLU()

        self._w = nn.Parameter(torch.Tensor(*(self.input_embed, self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._wl = nn.Parameter(torch.Tensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._wr = nn.Parameter(torch.Tensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._u = nn.Parameter(torch.Tensor(*(self.decoder_hidden_dim, 1)).to(self.device))
        self._b = nn.Parameter(torch.Tensor(*(self.decoder_hidden_dim, 1)).to(self.device))
        self._l = nn.Parameter(torch.Tensor(1).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._w)
        nn.init.xavier_uniform_(self._wl)
        nn.init.xavier_uniform_(self._wr)
        nn.init.xavier_uniform_(self._u)
        nn.init.xavier_uniform_(self._b)

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = torch.randn([1]).numpy()[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        nn.init.constant_(self._l, bias_initial_value)

    def forward(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]

        W = self._w
        W_l = self._wl
        W_r = self._wr
        U = self._u
        B = self._b

        # Compute linear output with shape (batch_size, max_length, max_length, decoder_hidden_dim)
        dot_l = torch.einsum('ijk, kl->ijl', encoder_output, W_l)
        dot_r = torch.einsum('ijk, kl->ijl', encoder_output, W_r)

        tiled_l = torch.Tensor.repeat(torch.unsqueeze(dot_l, dim=2), (1, 1, self.max_length, 1))
        tiled_r = torch.Tensor.repeat(torch.unsqueeze(dot_r, dim=1), (1, self.max_length, 1, 1))

        linear_sum = tiled_l + tiled_r

        # Compute bilinear product with shape (batch_size, max_length, max_length, decoder_hidden_dim)
        bilinear_product = torch.einsum('ijk, knl, imn->ijml', encoder_output, W, encoder_output)

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = self.activation(bilinear_product + linear_sum + B.view(self.decoder_hidden_dim))
        elif self.decoder_activation == 'relu':
            final_sum = self.activation(bilinear_product + linear_sum + B.view(self.decoder_hidden_dim))
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = bilinear_product + linear_sum + B.view(self.decoder_hidden_dim)
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        logits = torch.einsum('ijkl, l->ijk', final_sum, U.view(self.decoder_hidden_dim))  # Readability

        self.logit_bias = self._l

        if self.use_bias:    # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        self.samples = []
        self.mask = 0
        self.mask_scores = []
        self.entropy = []

        for i in range(self.max_length):
            position = torch.ones([encoder_output.shape[0]],
                                  device=self.device) * i
            position = position.long()
            # Update mask
            self.mask = torch.zeros((encoder_output.shape[0], self.max_length),
                                    device=self.device).scatter_(1, position.view(encoder_output.shape[0], 1), 1)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
            sampled_arr.requires_grad=True

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy


class SingleLayerDecoder(nn.Module):

    def __init__(self, batch_size, max_length, input_dimension, input_embed,
                 decoder_hidden_dim, decoder_activation, use_bias,
                 bias_initial_value, use_bias_constant, is_train, device=None):

        super().__init__()

        self.batch_size = batch_size    # batch size
        self.max_length = max_length    # input sequence length (number of cities)
        self.input_dimension = input_dimension
        self.input_embed = input_embed    # dimension of embedding space (actor)
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_activation = decoder_activation
        self.use_bias = use_bias
        self.bias_initial_value = bias_initial_value
        self.use_bias_constant = use_bias_constant
        self.device = device
        self.is_training = is_train

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            self.activation = nn.Tanh()
        elif self.decoder_activation == 'relu':
            self.activation = nn.ReLU()

        self._wl = nn.Parameter(torch.Tensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._wr = nn.Parameter(torch.Tensor(*(self.input_embed, self.decoder_hidden_dim)).to(self.device))
        self._u = nn.Parameter(torch.Tensor(*(self.decoder_hidden_dim, 1)).to(self.device))
        self._l = nn.Parameter(torch.Tensor(1).to(self.device))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._wl)
        nn.init.xavier_uniform_(self._wr)
        nn.init.xavier_uniform_(self._u)

        if self.bias_initial_value is None:  # Randomly initialize the learnable bias
            bias_initial_value = torch.randn([1]).numpy()[0]
        elif self.use_bias_constant:  # Constant bias
            bias_initial_value = self.bias_initial_value
        else:  # Learnable bias with initial value
            bias_initial_value = self.bias_initial_value

        nn.init.constant_(self._l, bias_initial_value)

    def forward(self, encoder_output):
        # encoder_output is a tensor of size [batch_size, max_length, input_embed]
        W_l = self._wl
        W_r = self._wr
        U = self._u

        dot_l = torch.einsum('ijk, kl->ijl', encoder_output, W_l)
        dot_r = torch.einsum('ijk, kl->ijl', encoder_output, W_r)

        tiled_l = torch.Tensor.repeat(torch.unsqueeze(dot_l, dim=2), (1, 1, self.max_length, 1))
        tiled_r = torch.Tensor.repeat(torch.unsqueeze(dot_r, dim=1), (1, self.max_length, 1, 1))

        if self.decoder_activation == 'tanh':    # Original implementation by paper
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'relu':
            final_sum = self.activation(tiled_l + tiled_r)
        elif self.decoder_activation == 'none':    # Without activation function
            final_sum = tiled_l + tiled_r
        else:
            raise NotImplementedError('Current decoder activation is not implemented yet')

        # final_sum is of shape (batch_size, max_length, max_length, decoder_hidden_dim)
        logits = torch.einsum('ijkl, l->ijk', final_sum, U.view(self.decoder_hidden_dim))  # Readability

        self.logit_bias = self._l

        if self.use_bias:  # Bias to control sparsity/density
            logits += self.logit_bias

        self.adj_prob = logits

        self.mask = 0
        self.samples = []
        self.mask_scores = []
        self.entropy = []

        for i in range(self.max_length):
            position = torch.ones([encoder_output.shape[0]],
                                  device=self.device) * i
            position = position.long()

            # Update mask
            self.mask = torch.zeros((encoder_output.shape[0], self.max_length),
                                    device=self.device).scatter_(1, position.view(encoder_output.shape[0], 1), 1)
            self.mask = self.mask.to(self.device)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
            sampled_arr.requires_grad=True

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy


class MultiheadAttention(nn.Module):
    
    def __init__(self, input_dimension, num_units=None, device=None):

        super().__init__()
        self.device = device

        # Linear projections
        # Q_layer = nn.Linear(in_features=input_dimension, out_features=num_units)
        self.Q_layer = nn.Sequential(nn.Linear(in_features=input_dimension,
                                               out_features=num_units),
                                     nn.ReLU()).to(self.device)
        self.K_layer = nn.Sequential(nn.Linear(in_features=input_dimension,
                                               out_features=num_units),
                                     nn.ReLU()).to(self.device)
        self.V_layer = nn.Sequential(nn.Linear(in_features=input_dimension,
                                               out_features=num_units),
                                     nn.ReLU()).to(self.device)

        # Normalize
        self.bn_layer = nn.BatchNorm1d(input_dimension).to(self.device)  # 传入通道数

    def forward(self, inputs, num_heads=16, dropout_rate=0.1, is_training=True):

        input_dimension = inputs.shape[1]
        inputs = inputs.permute(0,2,1)

        Q = self.Q_layer(inputs)  # [batch_size, seq_length, n_hidden]
        K = self.K_layer(inputs)  # [batch_size, seq_length, n_hidden]
        V = self.V_layer(inputs)  # [batch_size, seq_length, n_hidden]

        # Split and concat
        Q_ = torch.cat(torch.split(Q, int(input_dimension/num_heads), dim=2), dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
        K_ = torch.cat(torch.split(K, int(input_dimension/num_heads), dim=2), dim=0)  # [batch_size, seq_length, n_hidden/num_heads]
        V_ = torch.cat(torch.split(V, int(input_dimension/num_heads), dim=2), dim=0)  # [batch_size, seq_length, n_hidden/num_heads]

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute([0, 2, 1]))  # num_heads*[batch_size, seq_length, seq_length]
        
        # Scale
        outputs = outputs / (K_.shape[-1] ** 0.5)

        # Activation
        outputs = torch.softmax(outputs, dim=0)  # num_heads*[batch_size, seq_length, seq_length]

        # Dropouts
        outputs = F.dropout(outputs, p=dropout_rate, training=is_training)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # num_heads*[batch_size, seq_length, n_hidden/num_heads]
        
        # Restore shape
        outputs = torch.cat(torch.split(outputs, int(outputs.shape[0]/num_heads), dim=0), dim=2)  # [batch_size, seq_length, n_hidden]
        
        # Residual connection
        outputs = outputs + inputs  # [batch_size, seq_length, n_hidden]

        outputs = outputs.permute(0,2,1)
        
        # Normalize
        outputs = self.bn_layer(outputs)  # [batch_size, seq_length, n_hidden]

        return outputs


# Apply point-wise feed forward net to a 3d tensor with shape [batch_size, seq_length, n_hidden]
# Returns: a 3d tensor with the same shape and dtype as inputs
class FeedForward(nn.Module):

    def __init__(self, num_units=(2048, 512), device=None):

        super().__init__()
        self.device = device

        # Inner layer
        self.conv1 = nn.Conv1d(in_channels=num_units[1],
                               out_channels=num_units[0],
                               kernel_size=(1,),
                               bias=True).to(self.device)
        # Readout layer
        self.conv2 = nn.Conv1d(in_channels=num_units[0],
                               out_channels=num_units[1],
                               kernel_size=(1,),
                               bias=True).to(self.device)

        self.bn_layer = nn.BatchNorm1d(num_units[1]).to(self.device)  # 传入通道数


    def forward(self, inputs):

        outputs = self.conv1(inputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.relu(outputs)

        # Residual connection
        outputs = outputs + inputs

        outputs = self.bn_layer(outputs)  # [batch_size, seq_length, n_hidden]

        return outputs


class TransformerDecoder(nn.Module):
 
    def __init__(self, batch_size, max_length, hidden_dim,
                 num_heads, num_stacks, is_train, device=None):

        super().__init__()

        self.batch_size = batch_size  # batch size
        self.max_length = max_length  # input sequence length (number of cities)
        # input_dimension*2+1 # dimension of input, multiply 2 for expanding dimension to input complex value to tf, add 1 high priority token, 1 pointing
        self.input_dimension = hidden_dim
        self.input_embed = hidden_dim  # dimension of embedding space (actor)
        self.num_heads = num_heads
        self.num_stacks = num_stacks
        self.device = device
        self.is_training = is_train

        # self._emb_params = LayerParams(self, 'emb', self.device)
        self.emb = nn.Parameter(torch.Tensor(*(1, self.input_embed, self.input_embed)).to(self.device))

        self.reset_parameters()

        # Batch Normalization
        self.bn_layer = nn.BatchNorm1d(self.input_dimension).to(self.device)  # 传入通道数

        # conv1d
        self.conv1 = nn.Conv1d(in_channels=self.input_embed,
                               out_channels=self.max_length,
                               kernel_size=(1,),
                               bias=True).to(self.device)

        # attention
        self.multihead_attention = MultiheadAttention(self.input_dimension,
                                                      num_units=self.input_embed,
                                                      device=self.device)

        # feedforward
        self.feedforward = FeedForward(num_units=[self.input_embed, self.input_embed],
                                       device=self.device)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb)

    def forward(self, inputs):

        inputs = inputs.permute(0,2,1)
 
        # Embed input sequence
        W_embed = self.emb
        W_embed_ = W_embed.permute(2,1,0)
        self.embedded_input = F.conv1d(inputs, W_embed_, stride=1)

        # Batch Normalization
        self.enc = self.bn_layer(self.embedded_input)
        
        # Blocks
        for i in range(self.num_stacks): # num blocks
            ### Multihead Attention
            self.enc = self.multihead_attention(self.enc,
                                                num_heads=self.num_heads,
                                                dropout_rate=0.0,
                                                is_training=self.is_training)

            ### Feed Forward
            self.enc = self.feedforward(self.enc)

        # Readout layer
        self.adj_prob = self.conv1(self.enc)
        self.adj_prob = self.adj_prob.permute(0,2,1)

        ########################################
        ########## Initialize process ##########
        ########################################
        # Keep track of visited cities
        self.mask = 0
        self.mask_scores = []
        self.entropy = []
        self.samples = []

        inputs = inputs.permute(0,2,1)
        for i in range(self.max_length):
            position = torch.ones([inputs.shape[0]], device=self.device) * i
            position = position.long()
            # Update mask
            self.mask = torch.zeros((inputs.shape[0], self.max_length),
                                    device=self.device).scatter_(1, position.view(inputs.shape[0], 1), 1)

            masked_score = self.adj_prob[:,i,:] - 100000000.*self.mask
            prob = distr.Bernoulli(logits=masked_score)  # probs input probability, logit input log_probability

            sampled_arr = prob.sample()  # Batch_size, seqlenght for just one node
            sampled_arr.requires_grad=True

            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            self.entropy.append(prob.entropy())

        return self.samples, self.mask_scores, self.entropy
