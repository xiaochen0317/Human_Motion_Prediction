import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.partition import PoolingLayer
from relative_position import cal_ST_SPD


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, hidden_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.hidden_features = hidden_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, hidden_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, src, bias=None):
        # src: [B, N, C]
        # bias: [B, N, N]
        h = torch.matmul(src, self.W)  # [B, N, C] * [C, C'] -> [N, N, C']
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        if bias is not None:
            e += bias

        attention = self.softmax(e)
        # attention = self.dropout(attention)
        # h_prime = torch.matmul(attention, h)
        #
        # return h_prime, attention
        return attention


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, factor, attn_dropout=0.1):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.factor = factor
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.factor, k.transpose(1, 2))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(self.softmax(attn))
        output = torch.matmul(attn, v)
        return output, attn


class Attention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super(Attention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
        self.FC = nn.Linear(d_v, d_model, bias=False)
        self.attention = Scaled_Dot_Product_Attention(factor=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.LN = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        b_sz, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.W_Q(q).view(b_sz, len_q, d_k)
        k = self.W_K(k).view(b_sz, len_k, d_k)
        v = self.W_V(v).view(b_sz, len_v, d_v)
        if mask is not None:
            mask = mask.unqueeze(1)
        q, attn = self.attention(q, k, v, mask=mask)
        # attn: b_sz, len_q, len_k (len_k = len_v)
        # q: b_sz, len_q, d_v
        q = self.dropout(self.FC(q))  # q: b_sz, len_q, d_model
        q += residual
        # todo: LN应该在残差里还是外？
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


class Coarse_SpatialAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, joints, frames, spatial_scales, dropout=0.1):
        super(Coarse_SpatialAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.joints = joints
        self.frames = frames
        self.spatial_scales = spatial_scales

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.GraphAttn = nn.ModuleList()

        for i in range(len(spatial_scales)):
            self.GraphAttn.append(GraphAttentionLayer(in_features * frames, hidden_features))

        self.MLP1 = nn.ModuleList()
        self.MLP2 = nn.ModuleList()
        self.prelu = nn.ModuleList()

        for i in range(len(spatial_scales) - 1):
            self.MLP1.append(nn.Linear(spatial_scales[i], 2 * spatial_scales[i + 1], bias=True))
            self.MLP2.append(nn.Linear(2 * spatial_scales[i + 1], spatial_scales[i + 1], bias=True))
            self.prelu.append(nn.PReLU())

    def Spatial_Pooling(self, attn, i):
        S = self.MLP2[i](self.prelu[i](self.MLP1[i](attn)))
        return S

    def forward(self, src):
        # src: [B, T, J, C]
        attnList = []
        poolingList = []
        for i in range(len(self.spatial_scales)):
            attnList.append(self.GraphAttn[i](src))
            if i < len(self.spatial_scales) - 1:
                poolingList.append((self.Spatial_Pooling(attnList[i], i)))
                src = torch.matmul(poolingList[i].unsqueeze(1), src)
        return attnList, poolingList


class Fine_SpatialAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features1, hidden_features2,
                 joints, frames, spatial_scales, dropout=0.1):
        super(Fine_SpatialAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features1 = hidden_features1
        self.hidden_features2 = hidden_features2
        self.joints = joints
        self.frames = frames
        self.spatial_scales = spatial_scales

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.GraphAttn = nn.ModuleList()

        for i in range(len(spatial_scales)):
            self.GraphAttn.append(GraphAttentionLayer(in_features, hidden_features2))

        self.Coarse_Attention = Coarse_SpatialAttentionLayer(in_features, out_features, hidden_features1,
                                                             joints, frames, spatial_scales, dropout)

    def forward(self, src):
        # src: [B, T, J, C]
        attnList_Coarse, poolingList = self.Coarse_Attention(src)
        attnList_Fine = []
        for i in range(len(self.spatial_scales)):
            attnList_Fine.append(self.GraphAttn[i](src) + attnList_Coarse[i])
            if i < len(self.spatial_scales) - 1:
                src = torch.matmul(poolingList[i].unsqueeze(1), src)
        return attnList_Fine, poolingList


class Coarse_TemporalAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, joints, frames, temporal_scales, dropout=0.1):
        super(Coarse_TemporalAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.joints = joints
        self.frames = frames
        self.temporal_scales = temporal_scales

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.GraphAttn = nn.ModuleList()

        for i in range(len(temporal_scales)):
            self.GraphAttn.append(GraphAttentionLayer(in_features * joints, hidden_features))

        self.MLP1 = nn.ModuleList()
        self.MLP2 = nn.ModuleList()
        self.prelu = nn.ModuleList()

        for i in range(len(temporal_scales) - 1):
            self.MLP1.append(nn.Linear(temporal_scales[i], 2 * temporal_scales[i + 1], bias=True))
            self.MLP2.append(nn.Linear(2 * temporal_scales[i + 1], temporal_scales[i + 1], bias=True))
            self.prelu.append(nn.PReLU())

    def Spatial_Pooling(self, attn, i):
        S = self.MLP2[i](self.prelu[i](self.MLP1[i](attn)))
        return S

    def forward(self, src):
        # src: [B, T, J, C]
        attnList = []
        poolingList = []
        for i in range(len(self.temporal_scales)):
            attnList.append(self.GraphAttn[i](src))
            if i < len(self.temporal_scales) - 1:
                poolingList.append((self.Spatial_Pooling(attnList[i], i)))
                src = torch.matmul(poolingList[i].unsqueeze(1), src)
        return attnList, poolingList


# todo
class Fine_temporalAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features1, hidden_features2,
                 joints, frames, temporal_scales, dropout=0.1):
        super(Fine_temporalAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features1 = hidden_features1
        self.hidden_features2 = hidden_features2
        self.joints = joints
        self.frames = frames
        self.temporal_scales = temporal_scales

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.GraphAttn = nn.ModuleList()

        for i in range(len(temporal_scales)):
            self.GraphAttn.append(GraphAttentionLayer(in_features, hidden_features2))

        self.Coarse_Attention = Coarse_TemporalAttentionLayer(in_features, out_features, hidden_features1,
                                                              joints, frames, temporal_scales, dropout)

    def forward(self, src):
        # src: [B, T, J, C]
        attnList_Coarse, poolingList = self.Coarse_Attention(src)
        attnList_Fine = []
        for i in range(len(self.temporal_scales)):
            attnList_Fine.append(self.GraphAttn[i](src) + attnList_Coarse[i])
            if i < len(self.temporal_scales) - 1:
                src = torch.matmul(poolingList[i].unsqueeze(1), src)
        return attnList_Fine, poolingList


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
        self.edgefeat_linear1 = nn.Linear(hidden_features, hidden_features // 2, bias=False)
        self.prelu = nn.PReLU()
        self.edgefeat_linear2 = nn.Linear(hidden_features // 2, 1, bias=False)

    def forward(self, src, s_SPD, device='cuda:0'):  # src: B*T, J, C  s_SPD:List[J, J]
        B, N, C = src.size()
        edge_SPD_feat = torch.zeros([B, N, N, self.hidden_features]).to(device)
        for i in range(self.joints):
            for j in range(self.joints):
                SPD = s_SPD[i][j]
                # print('%d, %d'%(i,j))
                bone_num = len(SPD) - 1
                for k in range(bone_num):
                    head = SPD[k]
                    end = SPD[k]
                    edge = src[:, end, :] - src[:, head, :]  # bone_vector: B, C
                    edge = self.edge_emb_layer(edge)  # bone_vector: B, C'
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
        self.edgefeat_linear1 = nn.Linear(hidden_features, hidden_features // 2, bias=False)
        self.prelu = nn.PReLU()
        self.edgefeat_linear2 = nn.Linear(hidden_features // 2, 1, bias=False)

    def forward(self, src, t_SPD, device='cuda:0'):
        B, N, C = src.size()
        edge_SPD_feat = torch.zeros([B, N, N, self.hidden_features]).to(device)
        for i in range(self.frames):
            for j in range(self.frames):
                SPD = t_SPD[i][j]
                bone_num = len(SPD) - 1
                for k in range(bone_num):
                    head = SPD[k]
                    end = SPD[k]
                    edge = src[:, end, :] - src[:, head, :]  # bone_vector: B, C
                    edge = self.edge_emb_layer(edge)  # bone_vector: B, C'
                    edge_SPD_feat[:, i, j, :] += edge
        edge_attn = self.edgefeat_linear2(self.prelu(self.edgefeat_linear1(edge_SPD_feat)))
        return edge_attn


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

        # self.s_coef = Mix_Coefficient(spatial_scales[0], in_features, len(spatial_scales))
        # self.t_coef = Mix_Coefficient(temporal_scales[0], in_features, len(temporal_scales))

        self.softmax = nn.Softmax(dim=-1)

        self.spatialAttn = Fine_SpatialAttentionLayer(in_features, out_features, hidden_features, hidden_features,
                                                      spatial_scales[0], temporal_scales[0], spatial_scales)
        self.temporalAttn = Fine_temporalAttentionLayer(in_features, out_features, hidden_features, hidden_features,
                                                        spatial_scales[0], temporal_scales[0], temporal_scales)

    def forward(self, src, device='cuda:0'):
        B, C, T, J = src.size()
        # todo:s作为左右乘的元素？
        spatial_adj, _ = self.spatialAttn(src)
        temporal_adj, _ = self.temporalAttn(src)

        sa_fusion = spatial_adj[0]
        ta_fusion = temporal_adj[0]
        for i in range(1, len(spatial_adj)):
            sa_fusion += spatial_adj[i]
        for i in range(1, len(temporal_adj)):
            ta_fusion += temporal_adj[i]

        return sa_fusion, ta_fusion


class DMS_ST_GAT_layer(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, kernel_size, stride, heads, spatial_scales,
                 temporal_scales, attn_dropout=0.1, dropout=0.1):
        super(DMS_ST_GAT_layer, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size[0] % 2 == 1
        assert self.kernel_size[1] % 2 == 1
        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)

        # self.attention = DMS_STAttention(in_features, out_features, hidden_features, heads, spatial_scales,
        #                                  temporal_scales, attn_dropout)
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
        S, T = self.attention(x)  # [B, T, H, J, C_out] [B, T, H, J, J] [B, J, H, T, T]
        # x = x.permute(0, 3, 1, 2)
        # todo:改一下乘法
        out = torch.einsum('nctv,ntvw->nctw', (x, S))  # B T J C | B T J J  B T J C
        # out = torch.matmul(S, x)
        # out = torch.matmul(T, out.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = torch.einsum('nctv,nvtq->ncqv', (out, T))  # B T J C | B J T T  B T J C
        # out = torch.mean(out, dim=2)
        # out = out.permute(0, 3, 1, 2)  # B, C, T, J
        out = self.cnn(out)  # B, C, T, J
        out = self.prelu(out)
        out = out + res
        # out = out.permute(0, 2, 3, 1)
        return out


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
        self.st_gcnns.append(DMS_ST_GAT_layer(3 * in_features, 32, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, 64, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, 128, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(128, 64, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(64, 32, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))
        self.st_gcnns.append(DMS_ST_GAT_layer(32, in_features, hidden_features, [1, 1], 1, heads, spatial_scales,
                                              temporal_scales, st_attn_dropout, st_cnn_dropout))

        self.txcnns.append(TCN_Layer(input_time_frame, output_time_frame, txc_kernel_size, txc_dropout))
        # with kernel_size[3,3] the dimensinons of C,V will be maintained
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

        x = self.prelus[0](self.txcnns[0](x))
        temp2 = x

        for i in range(1, self.n_txcnn_layers):
            x = self.prelus[i](self.txcnns[i](x)) + x  # residual connection
        x = temp2 + x

        # x = x.permute(0, 2, 1, 3)
        # x = self.refine(x)
        # x = x.permute(0, 2, 1, 3)

        return x
