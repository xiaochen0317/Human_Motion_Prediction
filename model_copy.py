import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from fixed_adj import spatial_fixed_adj, temporal_fixed_adj
from utils.partition import PoolingLayer
import seaborn
from utils.Transformer_Layer import Decoder_Layer
from relative_position import cal_ST_SPD
from torch_geometric.nn import GATConv


# def get_sinusoid_encoding_table(n_position, d_hid):
#     def get_position_angle_vec(position):
#         return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
#
#     sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#     sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
#     sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
#     return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 1, T, d_hid


# class Temporal_Positional_Encoding(nn.Module):
#     def __init__(self, d_hid, n_position=200):
#         super(Temporal_Positional_Encoding, self).__init__()
#         self.register_buffer('pos_table', get_sinusoid_encoding_table(n_position, d_hid))
#
#     def forward(self, x):  # B, 3, T， J
#         p = self.pos_table[:, :x.size(1)] * 1000
#         return x + p
#
#
# class Spatial_Positional_Encoding(nn.Module):
#     def __init__(self, d_hid):
#         super(Spatial_Positional_Encoding, self).__init__()
#         self.d_hid = d_hid
#
#     def forward(self, x):  # B, J, 3
#         bs, joints, feat_dim = x.size()
#         temp = x[:, 8, :]
#         temp = temp.unsqueeze(1).repeat(1, joints, 1)
#         c = (torch.norm(x / 1000 - temp / 1000, dim=-1))
#         p = torch.exp(-c).unsqueeze(2)
#         return x + p


class AttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_features, num_node, edge_features=64, attn_dropout=0.1, negative_slope=0.2):
        super(AttentionLayer, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.edge_features = edge_features
        self.dropout = attn_dropout
        self.negative_slope = negative_slope
        self.num_node = num_node

        self.lin = nn.Linear(in_features, hidden_features, bias=False)
        self.a_src = nn.Parameter(torch.FloatTensor(hidden_features, 1))
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        self.a_dst = nn.Parameter(torch.FloatTensor(hidden_features, 1))
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

        self.lin_edge = nn.Linear(1, edge_features, bias=False)
        self.a_edge = nn.Parameter(torch.FloatTensor(edge_features, 1))
        nn.init.xavier_uniform_(self.a_edge, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        # self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.bias = None

    def forward(self, src, adj=None, mask=None, device='cuda:0'):
        # input: [B, N, in_features] central_point:[B, 1, in_features]
        B, N, _ = src.size()
        C = self.hidden_features

        x = self.lin(src)  # [B, N, C]
        x = self.leaky_relu(x)
        attn_src = torch.matmul(x, self.a_src)  # B, N, 1
        attn_dst = torch.matmul(x, self.a_dst)  # B, N, 1

        e = attn_src.expand(-1, -1, N) + attn_dst.expand(-1, -1, N).permute(0, 2, 1)

        if adj is not None:
            src1 = src.unsqueeze(2).repeat(1, 1, N, 1)  # B, N, N, C
            src2 = src.unsqueeze(2).repeat(1, 1, N, 1).permute(0, 2, 1, 3)  # B, N, N, C
            edge_attr_full = torch.norm(src1 - src2, 2, dim=-1)  # B, N, N, C

            edge_attr_full = torch.matmul(self.lin_edge(edge_attr_full.unsqueeze(-1)), self.a_edge).squeeze(-1)
            # B, N, N
            edge_attr_sparse = edge_attr_full * adj  # B, N, N
            e += edge_attr_sparse

        # a = self.leaky_relu(e)
        a = e

        if self.bias is not None:
            # todo: 加权
            a = e + self.bias
        if mask is not None:
            a = a * mask.to(x.dtype)

        attn = self.softmax(a)
        # attn = self.dropout(a)
        return attn  # [B, N, N] / [B, N+1, N+1]


class DMS_STAttention(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, spatial_scales, temporal_scales):
        super(DMS_STAttention, self).__init__()
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.spatial_gat = nn.ModuleList()
        self.spatial_pool = nn.ModuleList()
        self.temporal_gat = nn.ModuleList()
        self.temporal_pool = nn.ModuleList()

        self.sa_bias = nn.Parameter(torch.FloatTensor(1, 22, 22))
        nn.init.xavier_uniform_(self.sa_bias, gain=1.414)
        self.ta_bias = nn.Parameter(torch.FloatTensor(1, 10, 10))
        nn.init.xavier_uniform_(self.ta_bias, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)

        # self.dropout = nn.Dropout(0.1)

        for i in range(len(spatial_scales)):
            self.spatial_gat.append(AttentionLayer(in_features, hidden_features, spatial_scales[i]))
            if i < len(spatial_scales) - 1:
                self.spatial_pool.append(
                    PoolingLayer(self.in_features, 32, self.spatial_scales[i + 1], self.spatial_scales[i]))
        for i in range(len(temporal_scales)):
            self.temporal_gat.append(AttentionLayer(in_features, hidden_features, temporal_scales[i]))
            if i < len(temporal_scales) - 1:
                self.temporal_pool.append(
                    PoolingLayer(self.in_features, 32, self.temporal_scales[i + 1], self.temporal_scales[i]))

    def forward(self, src, device='cuda:0'):
        B, C, T, J = src.size()
        # todo:s作为左右乘的元素？
        s_ma1 = []
        s_ma2 = []
        sm_loss = 0
        so_loss = 0
        tm_loss = 0
        to_loss = 0
        spatial_edge_index = torch.tensor([
            [8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19,
             0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
             8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19]])
        temporal_edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8,
             1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9,
             0, 1, 2, 3, 4, 5, 6, 7, 8]])
        spatial_adj = torch.sparse_coo_tensor(spatial_edge_index, torch.ones(spatial_edge_index.shape[1]),
                                              torch.Size([J, J])).to_dense()
        spatial_adj = spatial_adj.expand(B * T, -1, -1).to(device)
        temporal_adj = torch.sparse_coo_tensor(temporal_edge_index, torch.ones(temporal_edge_index.shape[1]),
                                               torch.Size([T, T])).to_dense()
        temporal_adj = temporal_adj.expand(B * J, -1, -1).to(device)
        # Spatial GAT
        spatial_input = src.permute(0, 2, 3, 1).reshape(B * T, J, C)
        spatial_attention = []

        attn = self.spatial_gat[0](spatial_input)
        if attn.size()[1] > self.spatial_scales[0]:
            attn = attn[:, :self.spatial_scales[0], :self.spatial_scales[0]]
        spatial_attention.append(attn)

        for i in range(1, len(self.spatial_scales)):
            spatial_input, s1, spatial_adj, m_loss, o_loss = \
                self.spatial_pool[i - 1](spatial_input, spatial_attention[i - 1])
            sm_loss = sm_loss + m_loss
            so_loss = so_loss + o_loss
            s_ma1.append(s1)

            attn = self.spatial_gat[i](spatial_input)
            if attn.size()[1] > self.spatial_scales[i]:
                attn = attn[:, :self.spatial_scales[i], :self.spatial_scales[i]]
            spatial_attention.append(attn)  # B*T，J， J
        sa_fusion = spatial_attention[0]
        for i in range(1, len(self.spatial_scales)):
            A_prev = spatial_attention[i]
            S = s_ma1[0]
            if i >= 1:
                for j in range(1, i):
                    S = torch.matmul(S, s_ma1[j])
            A = S @ A_prev @ S.transpose(1, 2)
            sa_fusion = sa_fusion + A

        # Temporal GAT
        temporal_input = src.permute(0, 3, 2, 1).reshape(B * J, T, C)
        temporal_attention = []

        attn = self.temporal_gat[0](temporal_input)
        if attn.size()[1] > self.temporal_scales[0]:
            attn = attn[:, :self.temporal_scales[0], :self.temporal_scales[0]]
        temporal_attention.append(attn)  # B*T，J， J

        for i in range(1, len(self.temporal_scales)):
            temporal_input, s2, temporal_adj, m_loss, o_loss \
                = self.temporal_pool[i - 1](temporal_input, temporal_attention[i - 1])
            tm_loss = tm_loss + m_loss
            to_loss = to_loss + o_loss
            s_ma2.append(s2)

            attn = self.temporal_gat[i](temporal_input)
            if attn.size()[1] > self.temporal_scales[i]:
                attn = attn[:, :self.temporal_scales[i], :self.temporal_scales[i]]
            temporal_attention.append(attn)  # B*T，J， J
        ta_fusion = temporal_attention[0]
        for i in range(1, len(self.temporal_scales)):
            A_prev = temporal_attention[i]
            S = s_ma2[0]
            if i >= 1:
                for j in range(1, i):
                    S = torch.matmul(S, s_ma2[j])
            S = F.softmax(S, dim=-2)
            A = S @ A_prev @ S.transpose(1, 2)
            ta_fusion = ta_fusion + A

        sa_fusion = sa_fusion.reshape(B, T, J, J)
        ta_fusion = ta_fusion.reshape(B, J, T, T)
        # sa_fusion = self.softmax(sa_fusion)
        # ta_fusion = self.softmax(ta_fusion)
        sa_fusion = sa_fusion + self.sa_bias.unsqueeze(0).repeat(B, 1, 1, 1)
        ta_fusion = ta_fusion + self.ta_bias.unsqueeze(0).repeat(B, 1, 1, 1)
        # sa_fusion = self.dropout(sa_fusion)
        # ta_fusion = self.dropout(ta_fusion)

        return sa_fusion, ta_fusion, sm_loss, so_loss, tm_loss, to_loss


class DMS_ST_GAT_layer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, kernel_size, stride, heads, spatial_scales,
                 temporal_scales, attn_dropout=0.1, dropout=0.1):
        super(DMS_ST_GAT_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.attention = DMS_STAttention(in_features, out_features, hidden_features, spatial_scales,
                                         temporal_scales)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_features, out_features, (self.kernel_size[0], self.kernel_size[1]), (stride, stride), padding),
            nn.BatchNorm2d(out_features),
            nn.Dropout(dropout, inplace=True))

        if stride != 1 or in_features != out_features:
            self.residual = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=1, stride=(1, 1)),
                                          nn.BatchNorm2d(out_features))
        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        S, T, sm_loss, so_loss, tm_loss, to_loss = self.attention(x)

        out = torch.einsum('nctv,ntvw->nctw', (x, S))  # B T J C | B T J J  B T J C
        out = torch.einsum('nctv,nvtq->ncqv', (out, T))  # B T J C | B J T T  B T J C7
        out = self.cnn(out)
        out = out + res
        out = self.prelu(out)
        return out, sm_loss, so_loss, tm_loss, to_loss


class TCN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, bias=True):
        super(TCN_Layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
            (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)  # padding so that both dimensions are maintained
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                      nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, input_time_frame, output_time_frame, st_attn_dropout,
                 st_cnn_dropout, n_txcnn_layers, txc_kernel_size, txc_dropout, heads, spatial_scales,
                 temporal_scales, temporal_scales2):

        super(Model, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.st_gcnns = nn.ModuleList()
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()

        self.T_emb = nn.Parameter(torch.FloatTensor(3 * in_features, input_time_frame))
        self.S_emb = nn.Parameter(torch.FloatTensor(3 * in_features, spatial_scales[0]))
        nn.init.xavier_uniform_(self.T_emb, gain=1.414)
        nn.init.xavier_uniform_(self.S_emb, gain=1.414)

        self.st_gcnns.append(DMS_ST_GAT_layer(3 * in_features, 32, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        # self.st_gcnns.append(DMS_ST_GAT_layer(16, 32, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                       temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, 64, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, 128, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        # self.st_gcnns.append(DMS_ST_GAT_layer(128, 256, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                       temporal_scales, st_attn_dropout, st_cnn_dropout))
        # self.st_gcnns.append(DMS_ST_GAT_layer(256, 128, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                       temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(128, 64, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, 32, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        # self.st_gcnns.append(DMS_ST_GAT_layer(32, 16, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                       temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, in_features, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))

        self.txcnns.append(TCN_Layer(input_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(TCN_Layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()

        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

        self.residual = nn.Sequential(nn.Conv2d(input_time_frame, output_time_frame, kernel_size=1, stride=(1, 1)),
                                      nn.BatchNorm2d(output_time_frame))

        # self.refine = DMS_ST_GAT_layer(in_features, in_features, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                temporal_scales2, st_attn_dropout, st_cnn_dropout)

    def forward(self, x):
        e = 0
        l = 0
        B, C, T, J = x.size()
        temp1 = x
        sm_loss_all, so_loss_all, tm_loss_all, to_loss_all = 0, 0, 0, 0

        v = x[:, :, 1:] - x[:, :, :-1]
        v = torch.concat([v, v[:, :, -1].unsqueeze(2)], dim=2)
        a = v[:, :, 1:] - v[:, :, :-1]
        a = torch.concat([a, a[:, :, -1].unsqueeze(2)], dim=2)
        x = torch.concat([x, v, a], dim=1)
        x = x + self.S_emb.unsqueeze(0).unsqueeze(2) + self.T_emb.unsqueeze(0).unsqueeze(3)
        for gcn in self.st_gcnns:
            x, sm_loss, so_loss, tm_loss, to_loss = gcn(x)
            sm_loss_all, so_loss_all, tm_loss_all, to_loss_all = sm_loss_all + sm_loss, so_loss_all + so_loss, \
                                                                 tm_loss_all + tm_loss, to_loss_all + to_loss
        x = x + temp1

        x = x.permute(0, 2, 1, 3)

        temp2 = self.residual(x)
        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection
        x = temp2 + x

        return x, sm_loss_all, so_loss_all, tm_loss_all, to_loss_all
