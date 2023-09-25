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
from relative_position import cal_ST_SPD, cal_spatial_adj, cal_temporal_adj
from torch_geometric.nn import TransformerConv


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # 1, T, d_hid


class Temporal_Positional_Encoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(Temporal_Positional_Encoding, self).__init__()
        self.register_buffer('pos_table', get_sinusoid_encoding_table(n_position, d_hid))

    def forward(self, x):  # B, 3, T， J
        p = self.pos_table[:, :x.size(1)] * 1000
        return x + p


class Spatial_Positional_Encoding(nn.Module):
    def __init__(self, d_hid):
        super(Spatial_Positional_Encoding, self).__init__()
        self.d_hid = d_hid

    def forward(self, x):  # B, J, 3
        bs, joints, feat_dim = x.size()
        temp = x[:, 8, :]
        temp = temp.unsqueeze(1).repeat(1, joints, 1)
        c = (torch.norm(x / 1000 - temp / 1000, dim=-1))
        p = torch.exp(-c).unsqueeze(2)
        return x + p


class AttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_features, edge_features, heads=1, dropout=0.1, negative_slope=0.2,
                 bias=False, concat=True):
        super(AttentionLayer, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.heads = heads

        self.lin = nn.Linear(in_features, hidden_features, bias=False)
        self.a_src = nn.Parameter(torch.FloatTensor(hidden_features, heads))
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        self.a_dst = nn.Parameter(torch.FloatTensor(hidden_features, heads))
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

        self.lin_edge = nn.Linear(in_features, edge_features, bias=False)
        self.a_edge = nn.Parameter(torch.FloatTensor(edge_features, heads))
        nn.init.xavier_uniform_(self.a_edge, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * hidden_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(hidden_features))
        else:
            self.register_parameter('bias', None)

        # todo:在模块开始处加位置编码
        # if pos_enc == 'spatial':
        #     self.pos_enc = Spatial_Positional_Encoding(in_features)
        #     # self.adj = spatial_fixed_adj(joints, frames).to('cuda:0')
        # else:
        #     self.pos_enc = Temporal_Positional_Encoding(in_features)
        #     # self.adj = temporal_fixed_adj(joints, frames).to('cuda:0')

    def forward(self, src, edge_index=None, device='cuda:0'):  # input: [B, N, in_features]
        B, N, _ = src.size()
        H = self.heads
        C = self.hidden_features

        # central_point =

        # adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), torch.Size([N, N])).to_dense()
        # adj = adj.expand(B, -1, -1, H)

        x = self.lin(src)  # [B, N, C]
        attn_src = torch.matmul(x, self.a_src)  # B*T, J, H
        attn_dst = torch.matmul(x, self.a_dst)  # B*T, J, H

        attn = attn_src.unsqueeze(2).repeat(1, 1, N, 1) + attn_dst.unsqueeze(2).repeat(1, 1, N, 1).permute(0, 2, 1, 3)
        # if adj is not None:
        #     zero_vec = -9e15 * torch.ones_like(attn)
        #     attn = torch.where(adj > 0, attn, zero_vec)  # [N,N]

        # edge_attr_sparse = src[:, edge_index[0], :] - src[:, edge_index[1], :]  # B, E, C
        # edge_attr_sparse = self.lin_edge(edge_attr_sparse)
        # edge_attr_sparse = torch.matmul(edge_attr_sparse, self.a_edge)  # B, E, H
        # edge_attr_dense = torch.empty([B, N, N, H])
        #
        # for i in range(B):
        #     for j in range(H):
        #         edge_attr_dense[i, :, :, j] = torch.sparse_coo_tensor(edge_index.to(device), edge_attr_sparse[i, :, j],
        #                                                               torch.Size([N, N])).to_dense()
        #
        # edge_index_matrix = torch.sparse_coo_tensor(edge_index.to(device), 1.0, torch.Size([N, N])).to_dense()
        # edge_index_matrix = edge_index_matrix.to(device)
        # edge_attr_matrix = edge_attr_sparse.permute(0, 2, 1)  # B, H, E
        # edge_attr_dense = torch.matmul(edge_attr_matrix, edge_index_matrix).permute(0, 2, 1)  # B, N, H

        edge_attr_sparse = src[:, edge_index[0], :] - src[:, edge_index[1], :]  # B, E, C
        edge_attr_sparse = self.lin_edge(edge_attr_sparse)
        edge_attr_sparse = torch.matmul(edge_attr_sparse, self.a_edge)  # B, E, H

        # Construct a sparse tensor from edge_index and edge_attr_sparse
        edge_attr_coo = torch.sparse_coo_tensor(edge_index[0], edge_attr_sparse.view(B * edge_index.shape[1], H),
                                                torch.Size([B, N, N, H])).coalesce()

        # Convert the sparse tensor to dense
        edge_attr_dense = edge_attr_coo.to_dense()

        # Create edge_index_matrix
        edge_index_matrix = torch.sparse_coo_tensor(edge_index, 1.0, torch.Size([N, N])).to_dense()

        # Compute edge_attr_matrix using transpose
        edge_attr_matrix = edge_attr_sparse.permute(0, 2, 1)  # B, H, E

        # Perform matrix multiplication and transpose
        edge_attr_dense_matrix_mul = torch.matmul(edge_attr_matrix, edge_index_matrix).transpose(1, 2)  # B, N, H

        edge_attr_dense = edge_attr_dense.to(device)
        attn = attn + edge_attr_dense
        if self.bias is not None:
            # todo: 加权
            attn = attn + self.bias

        attn = self.leaky_relu(attn)
        attn = self.dropout(self.softmax(attn))

        return attn  # [B, N, N]


class Mix_Coefficient(nn.Module):
    def __init__(self, dim, channel, scales):
        super(Mix_Coefficient, self).__init__()
        self.linear1 = nn.Linear(channel, 1, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim, scales, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src):
        src = self.relu(self.linear1(src).squeeze(-1))
        src = self.linear2(src)
        # src = self.softmax(src)
        return src


class spatial_edge_enhanced_attention(nn.Module):
    def __init__(self, in_features, hidden_features, joints):
        super(spatial_edge_enhanced_attention, self).__init__()
        self.in_features = in_features  # C
        self.hidden_features = hidden_features  # C'
        self.joints = joints  # J

        self.edge_emb_layer = nn.Linear(in_features, hidden_features, bias=True)
        self.edgefeat_linear1 = nn.Linear(in_features, hidden_features // 2, bias=False)
        self.prelu = nn.PReLU()
        self.edgefeat_linear2 = nn.Linear(hidden_features // 2, 1, bias=False)

    def forward(self, src, s_SPD, device='cuda:0'):  # src: B*T, J, C  s_SPD:List[J, J]
        B, N, C = src.size()
        edge_SPD_feat = torch.zeros([B, N, N, C]).to(device)
        for i in range(self.joints):
            for j in range(self.joints):
                SPD = s_SPD[i][j]
                # print('%d, %d'%(i,j))
                bone_num = len(SPD) - 1
                for k in range(bone_num):
                    head = SPD[k]
                    end = SPD[k]
                    edge = src[:, end, :] - src[:, head, :]  # bone_vector: B, C
                    # edge = self.edge_emb_layer(edge)  # bone_vector: B, C'
                    edge_SPD_feat[:, i, j, :] += edge
        edge_attn = self.edgefeat_linear2(self.prelu(self.edgefeat_linear1(edge_SPD_feat)))
        return edge_attn


class temporal_edge_enhanced_attention(nn.Module):
    def __init__(self, in_features, hidden_features, frames):
        super(temporal_edge_enhanced_attention, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.frames = frames

        self.edge_emb_layer = nn.Linear(in_features, hidden_features, bias=True)
        self.edgefeat_linear1 = nn.Linear(in_features, hidden_features // 2, bias=False)
        self.prelu = nn.PReLU()
        self.edgefeat_linear2 = nn.Linear(hidden_features // 2, 1, bias=False)

    def forward(self, src, t_SPD, device='cuda:0'):
        B, N, C = src.size()
        edge_SPD_feat = torch.zeros([B, N, N, C]).to(device)
        for i in range(self.frames):
            for j in range(self.frames):
                SPD = t_SPD[i][j]
                bone_num = len(SPD) - 1
                for k in range(bone_num):
                    head = SPD[k]
                    end = SPD[k]
                    edge = src[:, end, :] - src[:, head, :]  # bone_vector: B, C
                    # edge = self.edge_emb_layer(edge)  # bone_vector: B, C'
                    edge_SPD_feat[:, i, j, :] += edge
        edge_attn = self.edgefeat_linear2(self.prelu(self.edgefeat_linear1(edge_SPD_feat)))
        return edge_attn


def calculate_edge_features(adjacency_matrix, node_features):
    # 计算边特征张量
    num_nodes = adjacency_matrix.size(0)
    batch_size = node_features.size(0)
    num_features = node_features.size(2)

    # 扩展邻接矩阵以适应节点特征集合的维度
    expanded_adj = adjacency_matrix.unsqueeze(2).expand(-1, -1, num_features)

    # 计算边特征张量
    edge_features = node_features.unsqueeze(1) - node_features.unsqueeze(2)
    edge_features *= expanded_adj

    return edge_features


class DMS_STAttention(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, spatial_scales, temporal_scales, heads=1):
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
        self.heads = heads

        # self.s_coef = Mix_Coefficient(spatial_scales[0], in_features, len(spatial_scales))
        # self.t_coef = Mix_Coefficient(temporal_scales[0], in_features, len(temporal_scales))

        self.sa_bias = nn.Parameter(torch.FloatTensor(temporal_scales[0], spatial_scales[0], spatial_scales[0]))
        nn.init.xavier_uniform_(self.sa_bias, gain=1.414)
        self.ta_bias = nn.Parameter(torch.FloatTensor(spatial_scales[0], temporal_scales[0], temporal_scales[0]))
        nn.init.xavier_uniform_(self.ta_bias, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)

        for i in range(len(spatial_scales)):
            self.spatial_gat.append(AttentionLayer(in_features, hidden_features, hidden_features))
            if i < len(spatial_scales) - 1:
                self.spatial_pool.append(
                    PoolingLayer(self.in_features, 32, self.spatial_scales[i + 1]))
        for i in range(len(temporal_scales)):
            self.temporal_gat.append(AttentionLayer(in_features, hidden_features, hidden_features))
            if i < len(temporal_scales) - 1:
                self.temporal_pool.append(
                    PoolingLayer(self.in_features, 32, self.temporal_scales[i + 1]))

        self.dropout = nn.Dropout(0.1)
        self.s_SPD, self.t_SPD = cal_ST_SPD()
        self.s_adj = torch.from_numpy(cal_spatial_adj()).to('cuda:0')
        self.t_adj = torch.from_numpy(cal_temporal_adj()).to('cuda:0')
        self.s_bias_emb = nn.Linear(in_features, 1)
        self.t_bias_emb = nn.Linear(in_features, 1)

    def forward(self, src, device='cuda:0'):
        B, C, T, J = src.size()
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
        # todo:s作为左右乘的元素？
        s_ma1 = []
        s_ma2 = []
        # Spatial GAT
        spatial_input = src.permute(0, 2, 3, 1).reshape(B * T, J, C)
        spatial_attention = []
        for i in range(len(self.spatial_scales) - 1):
            spatial_attention.append(self.spatial_gat[i](spatial_input, spatial_edge_index))  # B*T，J， J
            spatial_input, s1 = self.spatial_pool[i](spatial_input, spatial_attention[i])
            s_ma1.append(s1)
        spatial_attention.append(self.spatial_gat[-1](spatial_input, spatial_edge_index))
        sa_fusion = spatial_attention[0]
        for i in range(1, len(self.spatial_scales)):  # 从1尺度开始计算
            # 初始化邻接矩阵列表
            A_prev = spatial_attention[i]  # 每个尺度的注意力矩阵
            S = s_ma1[0]  # 初始化S为S(0,1)
            if i >= 1:  # 如果i大于1，那么需要计算S(0,i)
                for j in range(1, i):  # 对每个池化矩阵进行循环
                    # 从尺度i到尺度0，就需要S(0, i) = S(0, 1) * S(1, 2) *...* S(i-1, i)
                    S = torch.matmul(S, s_ma1[j])
            # 将结果添加到邻接矩阵列表中
            A = S @ A_prev @ S.transpose(1, 2)
            # S = F.softmax(S, dim=-2)
            sa_fusion += A
        # sa_fusion = sa_fusion / len(self.spatial_scales)
        # sa_fusion = torch.where(sa_fusion < 0.75, torch.zeros_like(sa_fusion), sa_fusion)

        # Temporal GAT
        temporal_input = src.permute(0, 3, 2, 1).reshape(B * J, T, C)
        temporal_attention = []
        for i in range(len(self.temporal_scales) - 1):
            temporal_attention.append(self.temporal_gat[i](temporal_input))
            temporal_input, s2 = self.temporal_pool[i](temporal_input, temporal_attention[i])
            s_ma2.append(s2)
        temporal_attention.append(self.temporal_gat[-1](temporal_input, temporal_edge_index))
        ta_fusion = temporal_attention[0]
        # ta_fusion = temporal_attention[0]
        for i in range(1, len(self.temporal_scales)):
            A_prev = temporal_attention[i]
            S = s_ma2[0]
            if i >= 1:
                for j in range(1, i):
                    S = torch.matmul(S, s_ma2[j])
            S = F.softmax(S, dim=-2)
            A = S @ A_prev @ S.transpose(1, 2)
            ta_fusion += A

        sa_fusion = sa_fusion.reshape(B, T, J, J, self.heads)
        # sa_fusion_avg = sa_fusion. sum(dim=1)
        ta_fusion = ta_fusion.reshape(B, J, T, T, self.heads)
        # ta_fusion_avg = ta_fusion.sum(dim=1)
        # sa_fusion += sa_fusion_avg.unsqueeze(1)
        # ta_fusion += ta_fusion_avg.unsqueeze(1)
        sa_fusion += self.sa_bias.unsqueeze(0).unsqueeze(4).repeat(B, 1, 1, 1, self.heads)
        ta_fusion += self.ta_bias.unsqueeze(0).unsqueeze(4).repeat(B, 1, 1, 1, self.heads)

        return sa_fusion, ta_fusion


class DMS_ST_GAT_layer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, kernel_size, stride, spatial_scales,
                 temporal_scales, heads=1, attn_dropout=0.1, dropout=0.1):
        super(DMS_ST_GAT_layer, self).__init__()
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.heads = heads
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.attention = DMS_STAttention(in_features, out_features, hidden_features, spatial_scales,
                                         temporal_scales)
        self.cnn = nn.ModuleList()
        for i in range(heads):
            self.cnn.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, (self.kernel_size[0], self.kernel_size[1]), (stride, stride), padding),
                nn.BatchNorm2d(out_features),
                nn.Dropout(dropout, inplace=True)))

        if stride != 1 or in_features != out_features:
            self.residual = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=1, stride=(1, 1)),
                                          nn.BatchNorm2d(out_features))
        else:
            self.residual = nn.Identity()

        self.prelu = nn.PReLU()

    def forward(self, x, device='cuda:0'):
        res = self.residual(x)
        B, C, T, J = x.size()
        S_attn, T_attn = self.attention(x)  # [B, T, H, J, C_out] [B, T, H, J, J] [B, J, H, T, T]
        out = torch.empty([B, self.out_features, T, J, self.heads])
        # todo:改一下乘法
        for i in range(self.heads):
            temp = torch.einsum('nctv,ntvw->nctw', (x, S_attn[:, :, :, :, i]))  # B C T J | B T J J  B C T J
            temp = torch.einsum('nctv,nvtq->ncqv', (temp, T_attn[:, :, :, :, i]))  # B C T J | B J T T  B C T J
            temp = self.cnn[i](temp)  # B, C, T, J
            out[:, :, :, :, i] = self.prelu(temp)
        out = torch.mean(out, dim=-1).to(device)
        out = out + res
        return out


class TCN_Layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
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
        self.st_gcnns.append(DMS_ST_GAT_layer(3 * in_features, 32, hidden_features, [1, 1], 1, spatial_scales,
                                              temporal_scales, heads, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, 64, hidden_features, [1, 1], 1, spatial_scales,
                                              temporal_scales, heads, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, 32, hidden_features, [1, 1], 1, spatial_scales,
                                              temporal_scales, heads, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, in_features, hidden_features, [1, 1], 1, spatial_scales,
                                              temporal_scales, heads, st_attn_dropout, st_cnn_dropout))

        self.txcnns.append(TCN_Layer(input_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(TCN_Layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()

        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

        self.residual = nn.Sequential(nn.Conv2d(input_time_frame, output_time_frame, kernel_size=1, stride=(1, 1)),
                                      nn.BatchNorm2d(output_time_frame))

        # self.st_gcnns2 = nn.ModuleList()
        # self.txcnns2 = nn.ModuleList()
        # self.T_emb2 = nn.Parameter(torch.FloatTensor(3 * in_features, output_time_frame))
        # self.S_emb2 = nn.Parameter(torch.FloatTensor(3 * in_features, spatial_scales[0]))
        # nn.init.xavier_uniform_(self.T_emb2, gain=1.414)
        # nn.init.xavier_uniform_(self.S_emb2, gain=1.414)
        # self.st_gcnns2.append(DMS_ST_GAT_layer(3 * in_features, 32, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                        temporal_scales2, st_attn_dropout, st_cnn_dropout))
        # self.st_gcnns2.append(DMS_ST_GAT_layer(32, 64, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                        temporal_scales2, st_attn_dropout, st_cnn_dropout))
        # self.st_gcnns2.append(DMS_ST_GAT_layer(64, 32, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                        temporal_scales2, st_attn_dropout, st_cnn_dropout))
        # self.st_gcnns2.append(DMS_ST_GAT_layer(32, in_features, hidden_features, [1, 1], 1, heads, spatial_scales,
        #                                        temporal_scales2, st_attn_dropout, st_cnn_dropout))

        # self.txcnns2.append(TCN_Layer(output_time_frame, input_time_frame, txc_kernel_size, txc_dropout))
        # with kernel_size[3,3] the dimensinons of C,V will be maintained
        # for i in range(1, n_txcnn_layers):
        #     self.txcnns2.append(TCN_Layer(input_time_frame, input_time_frame, txc_kernel_size, txc_dropout))
        #
        # self.residual2 = nn.Sequential(nn.Conv2d(output_time_frame, input_time_frame, kernel_size=1, stride=(1, 1)),
        #                                nn.BatchNorm2d(input_time_frame))
        #
        # self.txcnns3 = nn.ModuleList()
        # self.txcnns3.append(TCN_Layer(input_time_frame, input_time_frame, txc_kernel_size, txc_dropout))
        # # with kernel_size[3,3] the dimensinons of C,V will be maintained
        # for i in range(1, n_txcnn_layers):
        #     self.txcnns3.append(TCN_Layer(input_time_frame, input_time_frame, txc_kernel_size, txc_dropout))

    def forward(self, x):
        # B, C, T, J = x.size()
        temp1 = x
        v = x[:, :, 1:] - x[:, :, :-1]
        v = torch.concat([v, v[:, :, -1].unsqueeze(2)], dim=2)
        a = v[:, :, 1:] - v[:, :, :-1]
        a = torch.concat([a, a[:, :, -1].unsqueeze(2)], dim=2)
        x = torch.concat([x, v, a], dim=1)
        x = x + self.S_emb.unsqueeze(0).unsqueeze(2) + self.T_emb.unsqueeze(0).unsqueeze(3)
        for gcn in self.st_gcnns:
            x = gcn(x)
        x = x + temp1

        x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)
        temp = x
        temp2 = self.residual(x)

        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection
        x = temp2 + x

        # x = x.flip(dims=[1]).permute(0, 2, 1, 3)  # B, T, C, J -> B, C, T, J
        # temp3 = x
        # v = x[:, :, 1:] - x[:, :, :-1]
        # v = torch.concat([v, v[:, :, -1].unsqueeze(2)], dim=2)
        # a = v[:, :, 1:] - v[:, :, :-1]
        # a = torch.concat([a, a[:, :, -1].unsqueeze(2)], dim=2)
        # x = torch.concat([x, v, a], dim=1)
        # x = x + self.S_emb2.unsqueeze(0).unsqueeze(2) + self.T_emb2.unsqueeze(0).unsqueeze(3)
        # for gcn in self.st_gcnns2:
        #     x = gcn(x)
        # x = x + temp3
        #
        # x = x.permute(0, 2, 1, 3)
        # temp4 = self.residual2(x)
        #
        # x = self.prelus[0](self.txcnns2[0](x))
        # for i in range(1, self.n_txcnn_layers):
        #     x = self.prelus[i](self.txcnns2[i](x)) + x  # residual connection
        # x += temp4
        # neg = x.flip(dims=[1])
        #
        # offset = neg - temp1.permute(0, 2, 1, 3)
        # feedback = self.prelus[0](self.txcnns3[0](offset))
        #
        # for i in range(1, self.n_txcnn_layers):
        #     feedback = self.prelus[i](self.txcnns3[i](feedback)) + feedback
        #
        # x = temp + feedback
        # temp5 = self.residual(x)
        #
        # x = self.prelus[0](self.txcnns[0](x))
        #
        # for i in range(1, self.n_txcnn_layers):
        #     x = self.prelus[i](self.txcnns[i](x)) + x
        # x = temp5 + x

        return x
