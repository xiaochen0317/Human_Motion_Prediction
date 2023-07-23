import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from fixed_adj import spatial_fixed_adj, temporal_fixed_adj
from utils.partition import DiffPoolingLayer
import seaborn
from utils.Transformer_Layer import Decoder_Layer


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
    def __init__(self, in_features, hidden_features, alpha, dropout=0.1, negative_slope=0.2):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.alpha = alpha
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_features, hidden_features, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, 1, hidden_features))
        nn.init.normal_(self.att_src, mean=0, std=math.sqrt(hidden_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, 1, hidden_features))
        nn.init.normal_(self.att_dst, mean=0, std=math.sqrt(hidden_features))

        # if bias and concat:
        #     self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        # elif bias and not concat:
        #     self.bias = nn.Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)
        self.bias = None

        # todo:在模块开始处加位置编码
        # if pos_enc == 'spatial':
        #     self.pos_enc = Spatial_Positional_Encoding(in_features)
        #     # self.adj = spatial_fixed_adj(joints, frames).to('cuda:0')
        # else:
        #     self.pos_enc = Temporal_Positional_Encoding(in_features)
        #     # self.adj = temporal_fixed_adj(joints, frames).to('cuda:0')

    def forward(self, src, mask=None):  # input: [B, N, in_features]
        B, N, _ = src.size()
        C = self.hidden_features

        x = self.lin(src)  # [B, N, C]
        x = F.leaky_relu(x, self.negative_slope)

        attention1 = torch.sum(x * self.att_src, dim=-1)  # [B, N]
        attention2 = torch.sum(x * self.att_dst, dim=-1)  # [B, N]

        attention = attention1.unsqueeze(1) + attention2.unsqueeze(2)  # [B, N, N]
        attention = attention.softmax(dim=1)
        # attention = torch.where(attention < 0.75, torch.zeros_like(attention), attention)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        if self.bias is not None:
            # todo: 加权
            attention = attention + self.bias

        if mask is not None:
            attention = attention * mask.to(x.dtype)

        return attention  # [B, N, N]


class DMS_STAttention(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, alpha, spatial_scales, temporal_scales):
        super(DMS_STAttention, self).__init__()
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales
        self.alpha = alpha
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.spatial_gat = nn.ModuleList()
        self.spatial_pool = nn.ModuleList()
        self.temporal_gat = nn.ModuleList()
        self.temporal_pool = nn.ModuleList()

        # self.spatial_left = []
        # self.spatial_right = []
        # self.temporal_left = []
        # self.temporal_right = []

        for i in range(len(spatial_scales)):
            self.spatial_gat.append(AttentionLayer(in_features, hidden_features, alpha))
            if i < len(spatial_scales) - 1:
                self.spatial_pool.append(
                    DiffPoolingLayer(self.in_features, self.hidden_features, self.spatial_scales[i + 1]))
                # self.spatial_left.append(nn.Parameter(torch.FloatTensor(1, spatial_scales[0], spatial_scales[i + 1])))
                # stdv = 1. / math.sqrt(self.spatial_left[i].size(1))
                # self.spatial_left[i].data.uniform_(-stdv, stdv)
                # self.spatial_right.append(nn.Parameter(torch.FloatTensor(1, spatial_scales[i + 1], spatial_scales[0])))
                # stdv = 1. / math.sqrt(self.spatial_right[i].size(1))
                # self.spatial_right[i].data.uniform_(-stdv, stdv)
        for i in range(len(temporal_scales)):
            self.temporal_gat.append(AttentionLayer(in_features, hidden_features, alpha))
            if i < len(temporal_scales) - 1:
                self.temporal_pool.append(
                    DiffPoolingLayer(self.in_features, self.hidden_features, self.temporal_scales[i + 1]))
                # self.temporal_left.append(nn.Parameter(torch.FloatTensor(1, temporal_scales[0], temporal_scales[i + 1])))
                # stdv = 1. / math.sqrt(self.temporal_left[i].size(1))
                # self.temporal_left[i].data.uniform_(-stdv, stdv)
                # self.temporal_right.append(nn.Parameter(torch.FloatTensor(1, temporal_scales[i + 1], temporal_scales[0])))
                # stdv = 1. / math.sqrt(self.temporal_right[i].size(1))
                # self.temporal_right[i].data.uniform_(-stdv, stdv)

    def forward(self, src, device='cuda:0'):
        B, C, T, J = src.size()
        # todo:s作为左右乘的元素？
        s_ma1 = []
        s_ma2 = []
        # Spatial GAT
        e = 0
        l = 0
        spatial_input = src.permute(0, 2, 3, 1).reshape(B * T, J, C)
        spatial_attention = []
        for i in range(len(self.spatial_scales) - 1):
            spatial_attention.append(self.spatial_gat[i](spatial_input))  # B*T，J， J
            spatial_input, s1, l1, e1 = self.spatial_pool[i](spatial_input, spatial_attention[i])
            l += l1
            e += e1
            s_ma1.append(s1)
        spatial_attention.append(self.spatial_gat[-1](spatial_input))
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
            sa_fusion += A
        sa_fusion = sa_fusion.softmax(dim=1)
        # sa_fusion = torch.where(sa_fusion < 0.75, torch.zeros_like(sa_fusion), sa_fusion)

        # Temporal GAT
        temporal_input = src.permute(0, 3, 2, 1).reshape(B * J, T, C)
        temporal_attention = []
        for i in range(len(self.temporal_scales) - 1):
            temporal_attention.append(self.temporal_gat[i](temporal_input))
            temporal_input, s2, l2, e2 = self.temporal_pool[i](temporal_input, temporal_attention[i])
            l += l2
            e += e2
            s_ma2.append(s2)
        temporal_attention.append(self.temporal_gat[-1](temporal_input))
        ta_fusion = temporal_attention[0]
        for i in range(1, len(self.temporal_scales)):  # 从1尺度开始计算
            # 初始化邻接矩阵列表
            A_prev = temporal_attention[i]  # 每个尺度的注意力矩阵
            S = s_ma2[0]  # 初始化S为S(0,1)
            if i >= 1:  # 如果i大于1，那么需要计算S(0,i)
                for j in range(1, i):  # 对每个池化矩阵进行循环
                    # 从尺度i到尺度0，就需要S(0, i) = S(0, 1) * S(1, 2) *...* S(i-1, i)
                    S = torch.matmul(S, s_ma2[j])
            # 将结果添加到邻接矩阵列表中
            A = S @ A_prev @ S.transpose(1, 2)
            ta_fusion += A
        ta_fusion = ta_fusion.softmax(dim=1)
        # ta_fusion = torch.where(ta_fusion < 0.75, torch.zeros_like(ta_fusion), ta_fusion)

        sa_fusion = sa_fusion.reshape(B, T, J, J)
        ta_fusion = ta_fusion.reshape(B, J, T, T)
        return sa_fusion, ta_fusion, l, e


class DMS_ST_GAT_layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, kernel_size, stride, dropout, alpha,
                 spatial_scales, temporal_scales, bias=True):
        super(DMS_ST_GAT_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.attention = DMS_STAttention(in_features, hidden_features, out_features, alpha, spatial_scales,
                                         temporal_scales)

        self.tcn = nn.Sequential(
            nn.Conv2d(in_features, out_features, (self.kernel_size[0], self.kernel_size[1]), (stride, stride), padding),
            nn.BatchNorm2d(out_features),
            nn.Dropout(dropout, inplace=True))

        if stride != 1 or in_features != out_features:
            self.residual = nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=1, stride=(1, 1)),
                                          nn.BatchNorm2d(out_features))

        else:
            self.residual = nn.Identity()

        # todo: 加个Layer_norm
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)
        self.prelu = nn.PReLU()

    def forward(self, x):
        res = self.residual(x)
        S, T, l, e = self.attention(x)  # [B, T, J, J] [B, J, T, T]
        # todo:改一下乘法
        x = torch.einsum('nctv,ntvw->nctw', (x, S))  # B 3 T J | B T J J  B 3 H T J
        x = torch.einsum('nctv,nvtq->ncqv', (x, T))  # B 3 H T J | B J T T  B 3 H T J
        x = self.tcn(x)  # B, C, T, J
        x = x + res
        x = self.prelu(x)
        return x, l, e


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_channels = [num_channels]
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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


class TransformerDecodingLayer(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dropout=0.1, activation="relu"):
        super(TransformerDecodingLayer, self).__init__()
        dim_feedforward = 2 * d_model
        self.transformer_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, activation,
                                                            batch_first=True)
        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers)

    def forward(self, tgt, memory, mask=None):
        tgt = self.transformer_layer(tgt, memory, mask)
        return tgt


class Model(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 heads,
                 alpha,
                 spatial_scales,
                 temporal_scales,
                 bias=True):

        super(Model, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.st_gcnns = nn.ModuleList()
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()

        self.st_gcnns.append(DMS_ST_GAT_layer(in_features, hidden_features, 32, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, hidden_features, 64, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, hidden_features, 32, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, hidden_features, in_features, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))

        self.txcnns.append(TCN_Layer(input_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(TCN_Layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()

        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        e = 0
        l = 0
        for gcn in self.st_gcnns:
            x, l1, e1 = gcn(x)
            e += e1
            l += l1

        x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection

        return x, l, e


class Model_Reverse(nn.Module):
    """
    Shape:
        - Input[0]: Input sequence in :math:`(N, in_channels,T_in, V)` format
        - Output[0]: Output sequence in :math:`(N,T_out,in_channels, V)` format
        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
            :in_channels=number of channels for the coordiantes(default=3)
            +
    """

    def __init__(self,
                 in_features,
                 hidden_features,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 heads,
                 d_model,
                 alpha,
                 spatial_scales,
                 temporal_scales,
                 bias=True):

        super(Model_Reverse, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.st_gcnns = nn.ModuleList()
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()

        self.st_gcnns.append(DMS_ST_GAT_layer(in_features, hidden_features, 32, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, hidden_features, 64, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, hidden_features, 128, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(128, hidden_features, 64, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, hidden_features, 32, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, hidden_features, in_features, [1, 1], 1, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))

        # self.txcnns.append(TCN_Layer(input_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        # with kernel_size[3,3] the dimensinons of C,V will be maintained
        # for i in range(1, n_txcnn_layers):
        #     self.txcnns.append(TCN_Layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        #
        # self.prelus = nn.ModuleList()
        #
        # for j in range(n_txcnn_layers):
        #     self.prelus.append(nn.PReLU())
        self.decoder.append(TransformerDecodingLayer(d_model=d_model, n_head=heads))

    def forward(self, x):
        e = 0
        l = 0
        for gcn in self.st_gcnns:
            x, l1, e1 = gcn(x)
            e += e1
            l += l1

        x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection

        return x, l, e