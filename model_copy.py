import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch_geometric.nn.dense import dense_gat_conv
from torch_geometric.nn.conv import GATConv, GATv2Conv
import matplotlib.pyplot as plt
from fixed_adj import spatial_fixed_adj, temporal_fixed_adj
from utils.partition import DiffPoolingLayer
import seaborn


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
        c = (torch.norm(x/1000 - temp/1000, dim=-1))
        p = torch.exp(-c).unsqueeze(2)
        return x + p


class AttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_features, heads, alpha, dropout=0.1, negative_slope=0.2):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.alpha = alpha
        self.heads = heads
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_features, hidden_features, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, 1, heads, hidden_features // heads))
        nn.init.normal_(self.att_src, mean=0, std=math.sqrt(hidden_features))
        self.att_dst = nn.Parameter(torch.Tensor(1, 1, heads, hidden_features // heads))
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
        H, C = self.heads, self.hidden_features // self.heads
        B, N, _ = src.size()

        x = self.lin(src).view(B, N, H, C)  # [B, N, H, C]

        attention1 = torch.sum(x * self.att_src, dim=-1)  # [B, N, H]
        attention2 = torch.sum(x * self.att_dst, dim=-1)  # [B, N, H]

        attention = attention1.unsqueeze(1) + attention2.unsqueeze(2)  # [B, N, N, H]
        attention = attention.permute(0, 3, 1, 2)  # [B, H, N, N]
        attention = F.leaky_relu(attention, self.negative_slope)
        attention = attention.softmax(dim=2)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        # attention = attention.mean(dim=-1)

        if self.bias is not None:
            # todo: 加权
            attention = attention + self.bias

        if mask is not None:
            attention = attention * mask.to(x.dtype)

        return attention  # [B, H, N, N]


class DMS_STAttention(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, heads, alpha, spatial_scales, temporal_scales):
        super(DMS_STAttention, self).__init__()
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales
        self.alpha = alpha
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.heads = heads

        self.spatial_gat = nn.ModuleList()
        self.spatial_pool = nn.ModuleList()
        self.temporal_gat = nn.ModuleList()
        self.temporal_pool = nn.ModuleList()

        self.spatial_left = []
        self.spatial_right = []
        self.temporal_left = []
        self.temporal_right = []

        for i in range(len(spatial_scales)):
            self.spatial_gat.append(AttentionLayer(in_features, hidden_features, heads, alpha))
            if i < len(spatial_scales) - 1:
                self.spatial_pool.append(
                    DiffPoolingLayer(self.in_features, self.hidden_features, self.spatial_scales[i + 1]))
                self.spatial_left.append(nn.Parameter(torch.FloatTensor(heads, spatial_scales[0], spatial_scales[i + 1])))
                stdv = 1. / math.sqrt(self.spatial_left[i].size(1))
                self.spatial_left[i].data.uniform_(-stdv, stdv)
                self.spatial_right.append(nn.Parameter(torch.FloatTensor(heads, spatial_scales[i + 1], spatial_scales[0])))
                stdv = 1. / math.sqrt(self.spatial_right[i].size(1))
                self.spatial_right[i].data.uniform_(-stdv, stdv)
        for i in range(len(temporal_scales)):
            self.temporal_gat.append(AttentionLayer(in_features, hidden_features, heads, alpha))
            if i < len(temporal_scales) - 1:
                self.temporal_pool.append(
                    DiffPoolingLayer(self.in_features, self.hidden_features, self.temporal_scales[i + 1]))
                self.temporal_left.append(nn.Parameter(torch.FloatTensor(heads, temporal_scales[0], temporal_scales[i + 1])))
                stdv = 1. / math.sqrt(self.temporal_left[i].size(1))
                self.temporal_left[i].data.uniform_(-stdv, stdv)
                self.temporal_right.append(nn.Parameter(torch.FloatTensor(heads, temporal_scales[i + 1], temporal_scales[0])))
                stdv = 1. / math.sqrt(self.temporal_right[i].size(1))
                self.temporal_right[i].data.uniform_(-stdv, stdv)

    def forward(self, src, device='cuda:0'):
        B, C, T, J = src.size()
        H = self.heads
        s_ma1 = []
        s_ma2 = []
        # Spatial GAT
        spatial_input = src.permute(0, 2, 3, 1).reshape(B * T, J, C)
        spatial_attention = []
        for i in range(len(self.spatial_scales) - 1):
            spatial_attention.append(self.spatial_gat[i](spatial_input))  # B*T， H， J， J
            spatial_input, s1, l1, e1 = self.spatial_pool[i](spatial_input, spatial_attention[i])
            s_ma1.append(s1)
        spatial_attention.append(self.spatial_gat[-1](spatial_input))
        spatial_attention_fusion = spatial_attention[0]
        for i in range(1, len(self.spatial_scales)):
            spatial_attention_fusion = spatial_attention_fusion + \
                                       torch.matmul(torch.matmul(self.spatial_left[i-1].unsqueeze(0).repeat(B*T, 1, 1, 1).to(device), spatial_attention[i]),
                                                    self.spatial_right[i - 1].unsqueeze(0).repeat(B*T, 1, 1, 1).to(device))
            # spatial_attention_fusion += torch.matmul(torch.matmul(s_ma1[i - 1], spatial_attention[i]),
            #                                          s_ma1[i - 1].permute(0, 1, 3, 2))
        # softmax

        # Temporal GAT
        temporal_input = src.permute(0, 3, 2, 1).reshape(B * J, T, C)
        temporal_attention = []
        for i in range(len(self.temporal_scales) - 1):
            temporal_attention.append(self.temporal_gat[i](temporal_input))
            temporal_input, s2, l2, e2 = self.temporal_pool[i](temporal_input, temporal_attention[i])
            s_ma2.append(s2)
        temporal_attention.append(self.temporal_gat[-1](temporal_input))
        temporal_attention_fusion = temporal_attention[0]
        for i in range(1, len(self.temporal_scales)):
            temporal_attention_fusion = temporal_attention_fusion + \
                             torch.matmul(torch.matmul(self.temporal_left[i - 1].unsqueeze(0).repeat(B*J, 1, 1, 1).to(device), temporal_attention[i]),
                                          self.temporal_right[i - 1].unsqueeze(0).repeat(B*J, 1, 1, 1).to(device))
            # temporal_attention_fusion += torch.matmul(torch.matmul(s_ma2[i - 1], temporal_attention[i]),
            #                                           s_ma2[i - 1].permute(0, 1, 3, 2))

        spatial_attention_fusion = spatial_attention_fusion.reshape(B, T, H, J, J)
        temporal_attention_fusion = temporal_attention_fusion.reshape(B, J, H, T, T)
        return spatial_attention_fusion, temporal_attention_fusion


class DMS_ST_GAT_layer(nn.Module):
    def __init__(self, in_features,hidden_features, out_features, kernel_size, stride, heads, dropout, alpha,
                 spatial_scales, temporal_scales, bias=True):
        super(DMS_ST_GAT_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.attention = DMS_STAttention(in_features, hidden_features, out_features, heads, alpha, spatial_scales,
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
        S, T = self.attention(x)  # [B, T, H, J, J] [B, J, H, T, T]
        # todo:改一下乘法
        x = torch.einsum('nctv,nthvw->nchtw', (x, S))  # B 3 T J | B T H J J  B 3 H T J
        x = torch.einsum('nchtv,nvhtq->nchqv', (x, T))  # B 3 H T J | B J H T T  B 3 H T J
        x = torch.mean(x, dim=2)  # B, 3, T, J
        x = self.tcn(x)  # B, C, T, J
        x = x + res
        x = self.prelu(x)
        return x


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
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

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

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            , nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


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

        self.st_gcnns.append(DMS_ST_GAT_layer(in_features, hidden_features, 32, [1, 1], 1, heads, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, hidden_features, 64, [1, 1], 1, heads, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, hidden_features, 32, [1, 1], 1, heads, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, hidden_features, in_features, [1, 1], 1, heads, st_gcnn_dropout,
                                              alpha, spatial_scales, temporal_scales))

        self.txcnns.append(TCN_Layer(input_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(TCN_Layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()

        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        for gcn in self.st_gcnns:
            x = gcn(x)

        x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection

        return x

