import torch
import torch.nn as nn
import math
import numpy as np
from utils.Transformer_Layer import Encoder_Layer
import torch.functional as F

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttentionLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(2*out_features, 1)

    def forward(self, x, adj):
        h = self.linear(x)
        a = self.attention(torch.cat([h.repeat(1, adj.size(1)).view(h.size(0), -1, h.size(1)), h.unsqueeze(1).repeat(1, adj.size(1), 1)], dim=-1))
        e = F.softmax(a, dim=1)
        output = torch.matmul(e.transpose(1, 2), h)
        return output.squeeze(1)


class GraphAttentionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GraphAttentionNetwork, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphAttentionLayer(input_size, hidden_size))
        for _ in range(num_layers-2):
            self.gat_layers.append(GraphAttentionLayer(hidden_size, hidden_size))
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, adj):
        for i in range(self.num_layers-1):
            x = F.elu(self.gat_layers[i](x, adj))
        output, _ = self.rnn(x)
        output = self.output_layer(output)
        return output



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
                 time_dim,
                 joints_dim,
                 d_model,
                 dims,
                 key
                 ):
        super(ConvTemporalGraphical, self).__init__()
        self.SAB = SAB(joints_dim, time_dim, d_model, dims, key)  # J, T, T
        self.TAB = TAB(time_dim, time_dim, d_model, dims, key)  # T, J, J
        self.A = nn.Parameter(torch.FloatTensor(time_dim, joints_dim,
           joints_dim))  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''

    def forward(self, x):
        # x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        # # x=self.prelu(x)
        # x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        # x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        A = self.SAB(x)
        T = self.TAB(x)
        x = torch.einsum('nctv,vntq->ncqv', (x, T))  # B 3 T J | J T T  B 3 T J
        x = torch.einsum('nctv,tnvw->nctw', (x, A))  # B 3 T J | T J J  B 3 T J
        return x.contiguous()




class ConvTemporalGraphical2(nn.Module):
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
                 time_dim,
                 joints_dim,
                 d_model,
                 dims,
                 key
                 ):
        super(ConvTemporalGraphical, self).__init__()
        self.SAB = SAB(joints_dim, time_dim, d_model, dims, key)  # J, T, T
        self.TAB = TAB(time_dim, time_dim, d_model, dims, key)  # T, J, J
        self.A = nn.Parameter(torch.FloatTensor(time_dim, joints_dim,
           joints_dim))  # learnable, graph-agnostic 3-d adjacency matrix(or edge importance matrix)
        stdv = 1. / math.sqrt(self.A.size(1))
        self.A.data.uniform_(-stdv, stdv)

        self.T = nn.Parameter(torch.FloatTensor(joints_dim, time_dim, time_dim))
        stdv = 1. / math.sqrt(self.T.size(1))
        self.T.data.uniform_(-stdv, stdv)
        '''
        self.prelu = nn.PReLU()
        
        self.Z=nn.Parameter(torch.FloatTensor(joints_dim, joints_dim, time_dim, time_dim)) 
        stdv = 1. / math.sqrt(self.Z.size(2))
        self.Z.data.uniform_(-stdv,stdv)
        '''

    def forward(self, x):
        # x = torch.einsum('nctv,vtq->ncqv', (x, self.T))
        # # x=self.prelu(x)
        # x = torch.einsum('nctv,tvw->nctw', (x, self.A))
        # x = torch.einsum('nctv,wvtq->ncqw', (x, self.Z))
        A = self.SAB(x)
        T = self.TAB(x)
        x = torch.einsum('nctv,vntq->ncqv', (x, T))  # B 3 T J | J T T  B 3 T J
        x = torch.einsum('nctv,tnvw->nctw', (x, A))  # B 3 T J | T J J  B 3 T J
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
                 time_dim,
                 joints_dim,
                 dropout,
                 d_model,
                 dim,
                 key,
                 bias=True):

        super(ST_GCNN_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        self.gcn = ConvTemporalGraphical(time_dim, joints_dim, d_model, dim, key)  # the convolution layer

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

        self.prelu = nn.PReLU()

    def forward(self, x):
        #   assert A.shape[0] == self.kernel_size[1], print(A.shape[0],self.kernel_size)
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.prelu(x)
        return x


class CNN_layer(
    nn.Module):  # This is the simple CNN layer,that performs a 2-D convolution while maintaining the dimensions of the input(except for the features dimension)

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

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            , nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


# In[11]:


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
                 joints_to_consider,
                 n_txcnn_layers,
                 txc_kernel_size,
                 txc_dropout,
                 d_model,
                 dim,
                 bias=True):

        super(Model, self).__init__()
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame
        self.joints_to_consider = joints_to_consider
        self.st_gcnns = nn.ModuleList()
        self.n_txcnn_layers = n_txcnn_layers
        self.txcnns = nn.ModuleList()

        self.st_gcnns.append(ST_GCNN_layer(input_channels, 32, [1, 1], 1, input_time_frame,
                                           joints_to_consider, st_gcnn_dropout, d_model, dim, 0))
        self.st_gcnns.append(ST_GCNN_layer(32, 64, [1, 1], 1, input_time_frame,
                                           joints_to_consider, st_gcnn_dropout, d_model, dim, 1))
        self.st_gcnns.append(ST_GCNN_layer(64, 32, [1, 1], 1, input_time_frame,
                                           joints_to_consider, st_gcnn_dropout, d_model, dim, 2))
        self.st_gcnns.append(ST_GCNN_layer(32, input_channels, [1, 1], 1, input_time_frame,
                                           joints_to_consider, st_gcnn_dropout, d_model, dim, 3))

        # at this point, we must permute the dimensions of the gcn network, from (N,C,T,V) into (N,T,C,V)
        self.txcnns.append(CNN_layer(input_time_frame, output_time_frame, txc_kernel_size,
                                     txc_dropout))  # with kernel_size[3,3] the dimensinons of C,V will be maintained
        for i in range(1, n_txcnn_layers):
            self.txcnns.append(CNN_layer(output_time_frame, output_time_frame, txc_kernel_size, txc_dropout))

        self.prelus = nn.ModuleList()
        for j in range(n_txcnn_layers):
            self.prelus.append(nn.PReLU())

    def forward(self, x):
        for gcn in (self.st_gcnns):
            x = gcn(x)

        x = x.permute(0, 2, 1, 3)  # prepare the input for the Time-Extrapolator-CNN (NCTV->NTCV)

        x = self.prelus[0](self.txcnns[0](x))

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection

        return x


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, factor, attn_dropout=0.1):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.factor, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)
        return output, attn


class self_attention(nn.Module):
    def __init__(self, factor, attn_dropout=0.1):
        super(self_attention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, mask=None):
        attn = torch.matmul(q / self.factor, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(self.softmax(attn))
        return attn


class multi_head_attention(nn.Module):
    def __init__(self, n_head, d_model, d_k, attn_dropout=0.1):
        super(multi_head_attention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.attn_dropout = attn_dropout
        self.W_Q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_head * d_k, bias=False)
        self.attention = self_attention(factor=d_k ** 0.5, attn_dropout=self.attn_dropout)

    def forward(self, q, k, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k = q.size(0), q.size(1), k.size(1)
        q = self.W_Q(q).view(sz_b, len_q, n_head, d_k)
        k = self.W_K(k).view(sz_b, len_k, n_head, d_k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        if mask is not None:
            mask = mask.unqueeze(1)
        attn = self.attention(q, k, mask=mask)
        return attn


class Multi_Head_Attention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(Multi_Head_Attention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.W_K = nn.Linear(d_model, n_head * d_k, bias=False)
        self.W_V = nn.Linear(d_model, n_head * d_v, bias=False)
        self.FC = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = Scaled_Dot_Product_Attention(factor=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # print(self.W_Q(q).size())
        q = self.W_Q(q).view(sz_b, len_q, n_head, d_k)
        k = self.W_K(k).view(sz_b, len_k, n_head, d_k)
        v = self.W_V(v).view(sz_b, len_v, n_head, d_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        # attn: sz_b, n_head, len_q, len_k (len_k = len_v)
        # q: sz_b, n_head, len_q, d_v
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.FC(q))
        q += residual
        q = self.LN(q)
        return q, attn


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(Position_wise_Feed_Forward, self).__init__()
        self.W_1 = nn.Linear(d_in, d_hid)
        self.W_2 = nn.Linear(d_hid, d_in)
        self.LN = nn.LayerNorm(d_in, eps=1e-6)
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.W_2(self.ReLU(self.W_1(x)))
        x = self.Dropout(x)
        x += residual
        x = self.LN(x)
        return x


class Encoder_Layer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Encoder_Layer, self).__init__()
        self.Self_Attention = Multi_Head_Attention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.Pos_FFN = Position_wise_Feed_Forward(d_model, d_inner, dropout=dropout)

    def forward(self, encoder_input, self_attn_mask=None):
        encoder_output, encoder_self_attn = self.Self_Attention(encoder_input, encoder_input, encoder_input,
                                                                mask=self_attn_mask)
        encoder_output = self.Pos_FFN(encoder_output)
        # return encoder_output, encoder_self_attn
        return encoder_output, encoder_self_attn


class Decoder_Layer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(Decoder_Layer, self).__init__()
        self.Self_Attn = Multi_Head_Attention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.Enc_Attn = Multi_Head_Attention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.Pos_FFN = Position_wise_Feed_Forward(d_model, d_inner, dropout=dropout)

    def forward(self, decoder_input, encoder_output, self_attn_mask=None, decoder_attn_mask=None):
        decoder_output, decoder_attn = self.Self_Attn(decoder_input, encoder_output, encoder_output,
                                                      mask=decoder_attn_mask)
        # decoder_output, decoder_attn2 = self.Enc_Attn(decoder_output, decoder_output, decoder_output,
        # mask=self_attn_mask)
        decoder_output = self.Pos_FFN(decoder_output)
        return decoder_output, None, decoder_attn
        # return decoder_output, decoder_attn2, decoder_attn


def get_pad_mask(seq, pad_idx):
    # todo  为什么unsqueeze？维度？
    return (seq != pad_idx).unsqueeze(-2)


class Temporal_Positional_Encoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(Temporal_Positional_Encoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # [B, T, d_model]

    def forward(self, x):
        p = self.pos_table[:, :x.size(1)].clone().detach()  # B, T, d_model
        return x + p


class Spatial_Positional_Encoding(nn.Module):
    def __init__(self, d_hid):
        super(Spatial_Positional_Encoding, self).__init__()
        self.d_hid = d_hid

    def forward(self, x):  # T, B, J, 3
        bs, joints = x.size(0), x.size(1)
        p = torch.zeros([bs, joints, self.d_hid]).to('cuda:0')

        for j in range(bs):
            central_point = x[j, 0, :]
            for k in range(joints):
                p[j, k, :] = x[j, k, :] - central_point
        # torch.exp()
        # p = p.data().detach()  # B, T, d_model
        return x + p


class SAB(nn.Module):
    def __init__(self, joints, time, d_model, dims, key):
        super(SAB, self).__init__()
        self.atom_encoder = nn.Linear(dims[key], d_model)
        self.model = Spatial_Graphormer(n_heads=8, d_hid=d_model, dropout=0.1, input_dropout=0.1,
                                        joints=joints, time=time)
        self.out_FFN = nn.Linear(d_model, joints)

    def forward(self, src, mask=None):
        x = src.clone()  # B, 3, J, T
        x = x.permute(2, 0, 3, 1)  # B, 3, T, J  -> T, B, J, 3

        x = self.atom_encoder(x)
        t_len, bs, s_len, d_hid = x.size()
        x = x.view(-1, s_len, d_hid)  # T, B, J, 3 -> T, B,
        # J, d_model
        adj = self.model.forward(x, spatial_mask=mask)
        adj = adj.view(t_len, bs, s_len, s_len)
        return adj


class TAB(nn.Module):
    def __init__(self, joints, time, d_model, dims, key):
        super(TAB, self).__init__()
        self.atom_encoder = nn.Linear(dims[key], d_model)
        self.model = Temporal_Graphormer(n_heads=8, d_hid=d_model, dropout=0.1, input_dropout=0.1,
                                         joints=joints, time=time)
        self.out_FFN = nn.Linear(d_model, time)

    def forward(self, src, mask=None):
        x = src.clone()
        x = x.permute(3, 0, 2, 1)  # B, 3, T, J -> J, B, T, 3
        x = self.atom_encoder(x)  # J, B, T, 3 -> J, B, T, d_model -> J*B, T, d_model
        s_len, bs, t_len, d_hid = x.size()
        x = x.view(-1, t_len, d_hid)
        # adj = None
        adj = self.model.forward(x, temporal_mask=mask)
        adj = adj.view(s_len, bs, t_len, t_len)
        return adj


class Temporal_Graphormer(nn.Module):
    def __init__(self, n_heads, d_hid, dropout, input_dropout, joints, time):
        super(Temporal_Graphormer, self).__init__()
        self.n_heads = n_heads
        self.d_hid = d_hid
        self.joints = joints
        self.time = time
        # self.apply(lambda module: init_parameters(module, n_layers=n_layers))
        # self.atom_encoder = nn.Embedding(64, d_hid, padding_idx=0)

        self.input_dropout = nn.Dropout(input_dropout)
        self.position_encoding = Temporal_Positional_Encoding(d_hid=d_hid)
        self.attn = multi_head_attention(n_head=n_heads, d_model=d_hid, d_k=d_hid // n_heads, attn_dropout=dropout)

    def forward(self, batched_data, temporal_mask=None):
        # batched_data: [batch_size, T, d_hid]
        x = batched_data.clone()
        # output = self.input_dropout(self.position_encoding(x))
        attn = self.attn(x, x)
        attn = torch.mean(attn, dim=1)
        return attn


class Spatial_Graphormer(nn.Module):
    def __init__(self, n_heads, d_hid, dropout, input_dropout, joints, time):
        super(Spatial_Graphormer, self).__init__()
        self.n_heads = n_heads
        self.d_hid = d_hid
        self.joints = joints
        self.time = time
        # self.apply(lambda module: init_parameters(module, n_layers=n_layers))
        self.position_encoding = Spatial_Positional_Encoding(d_hid=d_hid)
        # self.atom_encoder = nn.Embedding(64, d_hid, padding_idx=0)

        self.input_dropout = nn.Dropout(input_dropout)
        self.attn = multi_head_attention(n_head=n_heads, d_model=d_hid, d_k=d_hid // n_heads, attn_dropout=dropout)

    def forward(self, batched_data, spatial_mask=None):
        x = batched_data.clone()
        attn = self.attn(x, x)
        attn = torch.mean(attn, dim=1)
        return attn
