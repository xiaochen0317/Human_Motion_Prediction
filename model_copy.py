import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from fixed_adj import spatial_fixed_adj, temporal_fixed_adj
from utils.partition import partition_and_pooling


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


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, pos_enc, joints, frames, alpha, dropout=0.1, beta=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.beta = beta
        self.concat = concat
        self.joints = joints
        self.frames = frames

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.beta)

        if pos_enc == 'spatial':
            self.pos_enc = Spatial_Positional_Encoding(in_features)
            # self.adj = spatial_fixed_adj(joints, frames).to('cuda:0')
        else:
            self.pos_enc = Temporal_Positional_Encoding(in_features)
            # self.adj = temporal_fixed_adj(joints, frames).to('cuda:0')
            
    def forward(self, src, adj=None):  # input: [B, N, in_features]
        h = torch.matmul(self.pos_enc(src), self.W) # shape [B, N, out_features]
        B, N = h.size(0), h.size(1)
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=1)\
            .view(B, N, -1, 2 * self.out_features) # shape[N, N, 2*out_features]
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze())  # [N,N,1] -> [N,N]
        zero_vec = -9e15*torch.ones_like(e)
        if adj is None:
            adj = torch.ones_like(e)  # full-connected
        attention = torch.where(adj > 0, e, zero_vec)
        #todo: 加权
        # attention = attention + self.alpha * self.adj.unsqueeze(0)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        return attention


class DMS_STGAT(nn.Module):
    def __init__(self, in_features, hidden_features, joints, frames, alpha, spatial_scales, temporal_scales):
        super(DMS_STGAT, self).__init__()
        self.joints = joints
        self.frames = frames
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales
        self.alpha = alpha

        self.spatial_gat = nn.ModuleList()
        self.temporal_gat = nn.ModuleList()
        self.spatial_left = list()
        self.spatial_right = list()
        self.temporal_left = list()
        self.temporal_right = list()

        for i in range(len(spatial_scales)):
            self.spatial_gat.append(GATLayer(in_features, hidden_features, pos_enc='spatial', joints=spatial_scales[i],
                                             frames=frames, alpha=alpha))
            if i < len(spatial_scales) - 1:
                self.spatial_left.append(nn.Parameter(torch.FloatTensor(spatial_scales[i], spatial_scales[i + 1])))
                stdv = 1. / math.sqrt(self.spatial_left[i].size(1))
                self.spatial_left[i].data.uniform_(-stdv, stdv)
                self.spatial_right.append(nn.Parameter(torch.FloatTensor(spatial_scales[i + 1], spatial_scales[i])))
                stdv = 1. / math.sqrt(self.spatial_right[i].size(1))
                self.spatial_right[i].data.uniform_(-stdv, stdv)
        for i in range(len(temporal_scales)):
            self.temporal_gat.append(GATLayer(in_features, hidden_features, pos_enc='temporal', joints=joints,
                                              frames=temporal_scales[i], alpha=alpha))
            if i < len(temporal_scales) - 1:
                self.temporal_left.append(nn.Parameter(torch.FloatTensor(temporal_scales[i], temporal_scales[i + 1])))
                stdv = 1. / math.sqrt(self.temporal_left[i].size(1))
                self.temporal_left[i].data.uniform_(-stdv, stdv)
                self.temporal_right.append(nn.Parameter(torch.FloatTensor(temporal_scales[i + 1], temporal_scales[i])))
                stdv = 1. / math.sqrt(self.temporal_right[i].size(1))
                self.temporal_right[i].data.uniform_(-stdv, stdv)

        self.softmax = nn.Softmax(dim=-2)

    def forward(self, src):
        B, C, T, J = src.size()
        # Spatial GAT
        spatial_input = src.permute(0, 2, 3, 1).reshape(B * T, J, C)
        spatial_attention = []
        for i in range(len(self.spatial_scales)):
            spatial_attention.append(self.spatial_gat[i](spatial_input).view(B, T, self.spatial_scales[i],
                                                                             self.spatial_scales[i]))
            spatial_input = partition_and_pooling(spatial_input, spatial_attention[i], self.spatial_scales[i + 1],
                                                  mode='spatial')
        spatial_attention_fusion = spatial_attention[0]
        for i in range(1, len(self.spatial_scales)):
            spatial_attention_fusion = spatial_attention_fusion + self.spatial_left[i - 1] * spatial_attention[i] \
                                       * self.spatial_right[i - 1]
        # softmax
        spatial_attention_fusion = self.softmax(spatial_attention_fusion)

        # Temporal GAT
        temporal_input = src.permute(0, 3, 2, 1).reshape(B * J, T, C)
        temporal_attention = []
        for i in range(len(self.temporal_scales)):
            temporal_attention.append(self.temporal_gat[i](temporal_input).view(B, T, self.temporal_scales[i],
                                                                                self.temporal_scales[i]))
            temporal_input = partition_and_pooling(temporal_input, temporal_attention[i], self.temporal_scalesp[i + 1],
                                                   mode='temporal')
        temporal_attention_fusion = temporal_attention[0]
        for i in range(1, len(self.temporal_scales)):
            temporal_attention_fusion = temporal_attention_fusion + self.temporal_left[i - 1] * temporal_attention[i] \
                                       * self.temporal_right[i - 1]
        # softmax
        temporal_attention_fusion = self.softmax(temporal_attention_fusion)
        return spatial_attention_fusion, temporal_attention_fusion


# todo: multi-head
# class MultiHeadGATLayer(nn.Module):
#     def __init__(self, in_features, out_features, num_heads, dropout=0.1, alpha=0.2, concat=True):
#         super(MultiHeadGATLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_heads = num_heads
#         self.alpha = alpha
#         self.concat = concat
#
#         # Define separate linear transformations for each attention head
#         self.linear = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_heads)])
#         self.attention = nn.ModuleList([nn.Linear(2 * out_features, 1) for _ in range(num_heads)])
#
#         self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, input, adj):
#         # input: [B, N, in_features]
#         B, N = input.size(0), input.size(1)
#
#         # Apply linear transformation and concatenate outputs for each attention head
#         head_outputs = []
#         for i in range(self.num_heads):
#             h = self.linear[i](input)  # shape: [B, N, out_features]
#             a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=-1)  # shape: [B, N, N, 2 * out_features]
#             e = self.leakyrelu(self.attention[i](a_input)).squeeze(-1)  # shape: [B, N, N]
#             zero_vec = -9e15 * torch.ones_like(e)
#             attention = torch.where(adj > 0, e, zero_vec)
#             attention = F.softmax(attention, dim=2)
#             attention = F.dropout(attention, self.dropout)
#
#             # Apply attention weights to input features
#             h_prime = torch.matmul(attention, h)  # shape: [B, N, out_features]
#             head_outputs.append(h_prime)
#
#         # Concatenate or average the output from all attention heads
#         if self.concat:
#             output = torch.cat(head_outputs, dim=-1)  # shape: [B, N, num_heads * out_features]
#         else:
#             output = torch.mean(torch.stack(head_outputs), dim=0)  # shape: [B, N, out_features]
#
#         return output
#
#
# class MultiHeadGAT(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features, num_heads, num_layers, dropout=0.1, alpha=0.2, concat=True):
#         super(MultiHeadGAT, self).__init__()
#         self.dropout = dropout
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#
#         self.gat_layers = nn.ModuleList()
#         self.gat_layers.append(MultiHeadGATLayer(in_features, hidden_features, num_heads, dropout, alpha, concat))
#         for _ in range(num_layers - 1):
#             self.gat_layers.append(MultiHeadGATLayer(num_heads * hidden_features, hidden_features, num_heads, dropout, alpha, concat))
#         self.fc = nn.Linear(num_heads * hidden_features, out_features)
#
#     def forward(self, input, adj):
#         x = input
#         for layer in self.gat_layers:
#             x = layer(x, adj)
#             x = F.elu(x)  # Apply activation function after each GAT layer
#             x = F.dropout(x, self.dropout)
#         output = self.fc(x)
#         return output


class ConvTemporalGraphical(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.*
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Output: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 d_input,
                 d_hid,
                 d_output,
                 joints_dim,
                 time_dim,
                 alpha,
                 spatial_scales,
                 temporal_scales
                 ):
        super(ConvTemporalGraphical, self).__init__()
        self.STGAT = DMS_STGAT(d_input, d_hid, joints_dim, time_dim, alpha, spatial_scales, temporal_scales)
        # self.A = nn.Parameter(torch.FloatTensor(time_dim, joints_dim,
        #    joints_dim))  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        # stdv = 1. / math.sqrt(self.A.size(1))
        # self.A.data.uniform_(-stdv, stdv)
        #
        # self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        # stdv = 1. / math.sqrt(self.T.size(1))
        # self.T.data.uniform_(-stdv, stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''

    def forward(self, x):
        # x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        # # # x=self.prelu(x)
        # x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        # x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        # _, T, A = self.STGAT(x)
        T, A = self.STGAT(x)
        # temp = x
        x = torch.einsum('nctv,ntvw->nctw', (x, A))  # B 3 T J | B T J J  B 3 T J
        x = torch.einsum('nctv,nvtq->ncqv', (x, T))  # B 3 T J | B J T T  B 3 T J
        # AT = torch.einsum('ntvw,nvtq->ntvqw', (A, T))
        # x = torch.einsum('nctv,ntvqw->ncqw', (x, AT))
        # x = torch.einsum('nctv,ntvw->nctw', (x, A))  # B 3 T J | B T J J  B 3 T J
        # x = x+temp
        return x.contiguous()


class ST_GCNN_layer(nn.Module):
    """
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
            :in_channels= dimension of coordinates
            : out_channels=dimension of coordinates
            +
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 d_in,
                 d_hid, 
                 d_out,
                 dropout,
                 joints_dim,
                 time_dim,
                 alpha,
                 spatial_scales,
                 temporal_scales,
                 bias=True):
        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.gcn = ConvTemporalGraphical(d_in, d_hid, d_out, joints_dim, time_dim, alpha, spatial_scales,
                                         temporal_scales)

        self.tcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                (self.kernel_size[0], self.kernel_size[1]),
                (stride, stride),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(1, 1)),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.residual = nn.Identity()

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

        self.prelu = nn.PReLU()

    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.prelu(x)
        return x


class CNN_layer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout,
                 bias=True):
        super(CNN_layer, self).__init__()
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
                 input_channels,
                 input_time_frame,
                 output_time_frame,
                 st_gcnn_dropout,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 d_hid,
                 joints_dim,
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

        self.st_gcnns.append(ST_GCNN_layer(input_channels, 32, [1, 1], 1, input_channels, d_hid, 32, st_gcnn_dropout, joints_dim, input_time_frame, alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(ST_GCNN_layer(32, 64, [1, 1], 1, 32, d_hid, 64, st_gcnn_dropout, joints_dim, input_time_frame, alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(ST_GCNN_layer(64, 32, [1, 1], 1, 64, d_hid, 32, st_gcnn_dropout, joints_dim, input_time_frame, alpha, spatial_scales, temporal_scales))
        self.st_gcnns.append(ST_GCNN_layer(32, input_channels, [1, 1], 1, 32, d_hid, input_channels, st_gcnn_dropout, joints_dim, input_time_frame, alpha, spatial_scales, temporal_scales))

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(CNN_layer(input_time_frame, output_time_frame, txc_kernel_size,
                                     txc_dropout))  # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

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

