# coding = utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .sparsemax import Sparsemax

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 1.5)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if name.startswith("weight"):
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)


class SASA(nn.Module):
    """
    References
    ----------
    [1]: https://arxiv.org/abs/2205.03554
    """

    def __init__(self, max_len, segments_num, input_dim, class_num, h_dim,
                 dense_dim, drop_prob, lstm_layer, coeff, n_layer=1, thres=0.0, device=None):

        super(SASA, self).__init__()

        self.sparse_max = Sparsemax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.max_len = max_len
        self.segments_num = segments_num
        self.input_dim = input_dim
        self.class_num = class_num
        self.h_dim = h_dim
        self.dense_dim = dense_dim
        self.drop_prob = drop_prob
        self.lstm_layer = lstm_layer
        self.coeff = coeff
        self.n_gat_layer = n_layer

        self.thres = thres
        self.device = device

        self.layer_norm_list = [nn.LayerNorm(2 * h_dim).to(self.device) for _ in range(0, self.input_dim)]

        self.lstm_list = [
            nn.LSTM(input_size=1, hidden_size=self.h_dim, num_layers=self.lstm_layer, batch_first=True).to(self.device)
            for _ in range(0, self.input_dim)]

        self.regress_lstm_list = nn.ModuleList([nn.LSTM(input_size=1, hidden_size=self.h_dim,
                                                        num_layers=self.lstm_layer, batch_first=True)
                                                for _ in range(0, self.input_dim)]).to(self.device)

        self.self_attn_Q = nn.Sequential(nn.Linear(in_features=h_dim, out_features=h_dim),
                                         nn.ELU()
                                         ).to(self.device)
        self.self_attn_K = nn.Sequential(nn.Linear(in_features=h_dim, out_features=h_dim),
                                         nn.ELU()
                                         ).to(self.device)
        self.self_attn_V = nn.Sequential(nn.Linear(in_features=h_dim, out_features=h_dim),
                                         nn.ELU()
                                         ).to(self.device)

        self.mse = torch.nn.MSELoss().to(self.device)

        self.dropout_layer = nn.Dropout(drop_prob).to(self.device)

        self.regressive_layer = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.BatchNorm1d(self.h_dim),
            nn.LeakyReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(self.h_dim, 1)
        ).to(self.device)

        self.predict_layer = nn.Sequential(nn.BatchNorm1d(self.input_dim * 2 * h_dim),
                                           nn.Linear(self.input_dim * 2 * h_dim, dense_dim),
                                           nn.BatchNorm1d(dense_dim),
                                           nn.LeakyReLU(),
                                           nn.Dropout(drop_prob),
                                           nn.Linear(dense_dim, class_num))
        for one_lstm_layer in self.lstm_list:
            one_lstm_layer.apply(init_weights)
        self.self_attn_Q.apply(init_weights)
        self.self_attn_K.apply(init_weights)
        self.self_attn_V.apply(init_weights)
        self.predict_layer.apply(init_weights)


    def self_attention(self, Q, K, scale=True, sparse=True, k=3):
        """
        :param Q: [batch_size, segments_num, hidden_dim]
        :param K: [batch_size, segments_num, hidden_dim]
        :return: [batch_size, segments_num, segments_num]
        """

        segment_num = Q.shape[1]

        attention_weight = torch.bmm(Q, K.permute(0, 2, 1))
        attention_weight = torch.mean(attention_weight, dim=1, keepdim=True)

        if scale:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = k * torch.log(torch.tensor(segment_num, dtype=torch.float32)) * attention_weight

        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, segment_num]))
            attention_weight = torch.reshape(attention_weight_sparse, [-1, attention_weight.shape[1],
                                                                       attention_weight.shape[2]])
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def attention_fn(self, Q, K, scaled=False, sparse=True, k=1):
        segment_num = Q.shape[1]

        attention_weight = torch.bmm(F.normalize(Q, p=2, dim=-1), F.normalize(K, p=2, dim=-1).permute(0, 2, 1))

        if scaled:
            d_k = torch.tensor(K.shape[-1]).float()
            attention_weight = attention_weight / torch.sqrt(d_k)
            attention_weight = k * torch.log(torch.tensor(segment_num, dtype=torch.float32)) * attention_weight

        if sparse:
            attention_weight_sparse = self.sparse_max(torch.reshape(attention_weight, [-1, attention_weight.shape[-1]]))
            attention_weight = torch.reshape(attention_weight_sparse, [-1, attention_weight.shape[1], attention_weight.shape[2]])
        else:
            attention_weight = self.softmax(attention_weight)

        return attention_weight

    def mmd_loss(self, src_struct, tgt_struct, weight):
        delta = src_struct - tgt_struct
        loss_value = torch.mean(torch.matmul(delta, torch.transpose(delta, 1, 0))) * weight
        return loss_value

    def log(self, x, o=None, r=1.0):
        if x.is_cuda == True:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if o is None:
            o = torch.cat([torch.zeros(1, x.size(1) - 1), r * torch.Tensor([[1]])], dim=1).to(device)
        c = F.cosine_similarity(x, o, dim=1).view(-1, 1)
        theta = torch.acos(self.shrink(c))
        v = F.normalize(x - c * o, dim=1)[:, :-1]
        return r * theta * v

    def shrink(self, x, epsilon=1e-4):
        x[torch.abs(x) > (1 - epsilon)] = x[torch.abs(x) > (1 - epsilon)] * (1 - epsilon)
        return x

    def forward(self, x, labeled_sample_size, y_true, test_mode=False):
        """
         the shape of x is [batch_size, input_dim, segments_num, window_size, 1]
         window_size == max_length
        """
        hidden_state_list = {}
        intra_feature_context = {}
        self_attn_weight = {}
        domain_loss_alpha = []
        index_list = []

        for i in range(0, self.input_dim):
            # [batch_size, num_feat, num_segment, window_size, 1] -- > [batch_size, num_segment, window_size, 1] -- > [batch_size * num_segment, window_size, 1]
            univariate_x = torch.reshape(x[:, i, :, :, :], shape=[-1, self.max_len, 1])

            h_0 = Variable(torch.zeros((self.lstm_layer, univariate_x.shape[0], self.h_dim), device=x.device))
            c_0 = Variable(torch.zeros((self.lstm_layer, univariate_x.shape[0], self.h_dim), device=x.device))

            # h_out: [1, batch_size * num_segment, hidden_size]
            ula, (h_out, _) = self.lstm_list[i](univariate_x, (h_0, c_0))  # h_out is the final state

            h_out = F.normalize(F.relu(h_out), dim=-1)
            # [1, batch_size * num_segment, hidden_size] --> [batch_size, num_segment, hidden_size]
            final_hidden_state = torch.reshape(h_out, shape=[-1, self.segments_num,
                                                             self.h_dim])

            # [batch_size * segments_num, h_dim] -->[batch_size, segments_num, h_dim]
            hidden_state_list[i] = final_hidden_state

            Q = self.self_attn_Q(final_hidden_state)  # [batch_size, num_segment, hidden_size]
            K = self.self_attn_K(final_hidden_state)  # [batch_size, num_segment, hidden_size]
            V = self.self_attn_V(final_hidden_state)  # [batch_size, num_segment, hidden_size]

            attention_weight = self.self_attention(Q=Q, K=K, sparse=True)  # [batch_size, 1, segments_num]
            index = torch.argmax(attention_weight, dim=-1)
            index_list.append(index)

            Z_i = torch.bmm(attention_weight, V)  # [batch_size, 1, hidden_size]
            Z_i = F.normalize(Z_i, dim=-1)
            self_attn_weight[i] = attention_weight

            intra_feature_context[i] = Z_i

            src_strcuture, tgt_train_structure = torch.chunk(torch.squeeze(attention_weight), 2, dim=0)
            tgt_structure = tgt_train_structure

            align_src_structure = (src_strcuture > self.thres).float()
            align_tgt_structure = (tgt_structure > self.thres).float()

            domain_loss_intra = self.mmd_loss(src_struct=align_src_structure,
                                              tgt_struct=align_tgt_structure, weight=self.coeff)

            domain_loss_alpha.append(domain_loss_intra)

        final_feature = []
        attention_weight_graph_list = []
        domain_loss_beta = []
        for i in range(0, self.input_dim):
            Z_i = intra_feature_context[i]  # [batch_size, 1, hidden_dim]

            # [batch_size, num_feat, hidden_dim]
            other_hidden_state2 = torch.cat([hidden_state_list[j] for j in range(self.input_dim) if j!=i], dim=1)

            # [batch_size, num_feat].
            attention_weight = self.attention_fn(Q=Z_i, K=other_hidden_state2, sparse=True)
            attention_weight_graph = attention_weight > 0.0
            attention_weight_graph_list.append(torch.unsqueeze(attention_weight_graph.detach(), dim=1))

            # [batch_size, 1, hidden_dim]
            U_i = torch.bmm(attention_weight, other_hidden_state2)

            # [sample_size, 2 * input_dim]
            Hi = torch.squeeze(torch.cat([Z_i, U_i], dim=-1), dim=1)  # [batch_size, 2 * hidden_size]
            Hi = F.normalize(Hi, dim=-1)
            final_feature.append(Hi)

            src_strcuture, tgt_train_structure = torch.chunk(torch.squeeze(attention_weight), 2, dim=0)
            tgt_structure = tgt_train_structure

            align_src_structure = (src_strcuture > self.thres).float()
            align_tgt_structure = (tgt_structure > self.thres).float()

            domain_loss_inter = self.mmd_loss(src_struct=align_src_structure,
                                              tgt_struct=align_tgt_structure, weight=self.coeff)
            domain_loss_beta.append(domain_loss_inter)

        if test_mode == True:
            final_feature = torch.cat(final_feature, dim=-1)
        else:
            src_feature, tgt_train_feature = torch.chunk(torch.cat(final_feature, dim=-1), 2, dim=0)
            final_feature = torch.reshape(torch.cat((src_feature, tgt_train_feature)),
                                          shape=[labeled_sample_size, self.input_dim * 2 * self.h_dim])

        y_pred = self.predict_layer(final_feature)
        y_pred = torch.softmax(y_pred, dim=-1)

        if test_mode == False:
            label_loss = F.binary_cross_entropy(y_pred, y_true)

            total_loss, label_loss, total_domain_loss_alpha, total_domain_loss_beta = \
                self.calculate_loss(domain_loss_alpha=domain_loss_alpha,
                                    domain_loss_beta=domain_loss_beta,
                                    label_loss=label_loss)

            return y_pred, total_loss, label_loss, total_domain_loss_alpha, total_domain_loss_beta
        else:
            return y_pred

    def calculate_loss(self, domain_loss_alpha, domain_loss_beta, label_loss):
        total_domain_loss_alpha = torch.tensor(domain_loss_alpha).mean()
        total_domain_loss_beta = torch.tensor(domain_loss_beta).mean()
        total_loss = total_domain_loss_alpha + total_domain_loss_beta + label_loss

        return total_loss, label_loss, total_domain_loss_alpha, total_domain_loss_beta
