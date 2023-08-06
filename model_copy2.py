import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.partition import PoolingLayer
from relative_position import cal_ST_SPD


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
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
        if bias is not  None:
            e += bias

        attention = self.softmax(e)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)

        return h_prime, attention


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


class Spatial_Encoder(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, pooling_hidden_features,
                 joints, frames, spatial_scales, slope=0.2, dropout=0.1):
        super(Spatial_Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.pooling_hidden_features = pooling_hidden_features
        self.joints = joints
        self.frames = frames
        self.spatial_scales = spatial_scales
        self.all_attn1 = nn.Linear(frames * out_features, 1, bias=False)
        self.all_attn2 = nn.Linear(frames * out_features, 1, bias=False)

        self.QKV_emb()

        self.leaky_relu = nn.LeakyReLU(slope)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)


    def QKV_emb(self):
        self.W_Q = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.W_K = nn.Linear(self.in_features, self.hidden_features, bias=False)
        self.W_V = nn.Linear(self.in_features, self.hidden_features, bias=False)


    def all_GAT(self, src, bias=None):
        # src: [B, T, J, C_out]
        # bias: [B, J, J]
        B, T, J, C_in = src.size()
        src_emb = self.node_emb(src).permute(0, 2, 1, 3) # [B, T, J, C_out] -> [B, J, T, C_out]
        src_emb = src_emb.view(B, J, -1)  # [B, J, T*C_out]

        attn_src = self.all_attn1(src_emb)  # B, J, 1
        attn_tgr = self.all_attn1(src_emb)  # B, J, 1
        attention = attn_src.expand(-1, -1, J) + attn_tgr.expand(-1, -1, J).permute(0, 2, 1)  # B, J, J

        attention = self.leaky_relu(attention)
        if bias is not  None:
            attention += bias
        attention = self.dropout(attention)

        output = torch.matmul(attention, src_emb)  # [B, J, J] * [B, J, -1] -> [B, J, -1]

        return output, attention

    def Multiscale_Pooling(self, src, adj):
        # src: [B, T, J, C]
        # adj: [B, J, J]
        B, T, J, C_in = src.size()
        src =

        return S

    def forward(self, src):
        # src: [B, T, J, C]
        B, T, J, C_in = src.size()
        src_Q = self.W_Q(src)  # B, T, J, C_out
        src_K = self.W_K(src)  # B, T, J, C_out
        src_V = self.W_V(src)  # B, T, J, C_out



        return







class ST_GAT_Encoder(nn.Module):
    def __init__(self, joints, frames, in_features, out_features, node_emb_features, edge_emb_features,
                 spatial_scales, temporal_scales):
        super(ST_GAT_Encoder, self).__init__()
        self.joints = joints
        self.frames = frames
        self.in_features = in_features
        self.out_features = out_features
        self.node_emb_features = node_emb_features
        self.edge_emb_features = edge_emb_features

        self.fully_spatial_node_emb = nn.Linear(in_features, node_emb_features)
        self.fully_spatial_edge_emb = nn.Linear(in_features, edge_emb_features)
        self.fully_spatial_attention = nn.Parameter(torch.zeros(size=(node_emb_features+edge_emb_features, 1)))
        nn.init.xavier_uniform_(self.fully_spatial_attention.data, gain=1.414)
        self.fully_temporal_node_emb = nn.Linear(in_features, node_emb_features)
        self.fully_temporal_edge_emb = nn.Linear(in_features, edge_emb_features)


    def fully_spatial_multiscale_GAT(self, src):
        # src: [B, T, J, C_in]
        B, T, J, C_in = src.size()
        src = src.permute(0, 2, 1, 3)  # B, J, T, C_in
        node_feat = src.view(B, J, T * C_in)
        edge_feat = (src[:, :, 1:, :] - src[:, :, :-1, :]).view(B, J, (T - 1) * C_in)  # B, J, (T-1)*C_in
        node_feat_emb = self.fully_spatial_node_emb(node_feat)  # B, J, C_out1
        edge_feat_emb = self.fully_spatial_edge_emb(edge_feat)  # B, J, C_out2
        feat_emb = torch.concat([node_feat_emb, edge_feat_emb], dim=-1)  # B, J, C_out1+C_out2
        spatial_output, spatial_adj = self.fully_spatial_GAT(feat_emb)  # B, J, C_out  B, J, J





        return fully_spatial_output, fully_spatial_adj, spatial_pooling_mat


    def fully_multiscale_spatial_encoder(self, src):
        # src: [B, J, T*C]


        return


    def forward(self, src):
        # src: [B, T, J, C]
    class GAT(nn.Module):
        def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
            """Dense version of GAT."""
            super(GAT, self).__init__()
            self.dropout = dropout

            self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)]
            for i, attention in enumerate(self.attentions):
                self.add_module('attention_{}'.format(i), attention)

            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha,
                                               concat=False)  # 第二层(最后一层)的attention layer

        def forward(self, x, adj):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.attentions], dim=1)  # 将每层attention拼接
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))  # 第二层的attention layer
            return F.log_softmax(x, dim=1)

        def __init__(self, in_features, out_features, dropout, alpha, concat=True):
            super(GraphAttentionLayer, self).__init__()
            self.dropout = dropout
            self.in_features = in_features
            self.out_features = out_features
            self.alpha = alpha
            self.concat = concat

            self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
            self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # concat(V,NeigV)
            nn.init.xavier_uniform_(self.a.data, gain=1.414)

            self.leakyrelu = nn.LeakyReLU(self.alpha)
        return





















































class AttentionLayer(nn.Module):
    def __init__(self, in_features, hidden_features, dropout=0.1, negative_slope=0.2):
        super(AttentionLayer, self).__init__()
        # self.dropout = dropout
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_features, hidden_features, bias=False)
        self.a_src = nn.Parameter(torch.FloatTensor(hidden_features, 1))
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        self.a_dst = nn.Parameter(torch.FloatTensor(hidden_features, 1))
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.bias = None
        # todo:在模块开始处加位置编码

    def forward(self, src, mask=None):  # input: [B, N, in_features]
        B, N, _ = src.size()
        C = self.hidden_features

        x = self.lin(src)  # [B, N, C]
        attn_src = torch.matmul(x, self.a_src)  # B*T, J, 1
        attn_dst = torch.matmul(x, self.a_dst)  # B*T, J, 1
        attn = attn_src.expand(-1, -1, N) + attn_dst.expand(-1, -1, N).permute(0, 2, 1)
        attn = self.leaky_relu(attn)

        if self.bias is not None:
            # todo: 加权
            attn = attn + self.bias

        if mask is not None:
            attn = attn * mask.to(x.dtype)

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

        self.s_coef = Mix_Coefficient(spatial_scales[0], in_features, len(spatial_scales))
        self.t_coef = Mix_Coefficient(temporal_scales[0], in_features, len(temporal_scales))

        self.sa_bias = nn.Parameter(torch.FloatTensor(10, 22, 22))
        nn.init.xavier_uniform_(self.sa_bias, gain=1.414)
        self.ta_bias = nn.Parameter(torch.FloatTensor(22, 10, 10))
        nn.init.xavier_uniform_(self.ta_bias, gain=1.414)
        self.softmax = nn.Softmax(dim=-1)

        for i in range(len(spatial_scales)):
            self.spatial_gat.append(AttentionLayer(in_features, hidden_features))
            if i < len(spatial_scales) - 1:
                self.spatial_pool.append(
                    DiffPoolingLayer(self.in_features, 32, self.spatial_scales[i + 1]))
        for i in range(len(temporal_scales)):
            self.temporal_gat.append(AttentionLayer(in_features, hidden_features))
            if i < len(temporal_scales) - 1:
                self.temporal_pool.append(
                    DiffPoolingLayer(self.in_features, 32, self.temporal_scales[i + 1]))

        self.dropout = nn.Dropout(0.1)
        self.seea = spatial_edge_enhanced_attention(in_features, hidden_features, spatial_scales[0])
        self.teea = temporal_edge_enhanced_attention(in_features, hidden_features, temporal_scales[0])
        self.s_SPD, self.t_SPD = cal_ST_SPD()

    def forward(self, src, device='cuda:0'):
        B, C, T, J = src.size()
        # todo:s作为左右乘的元素？
        s_ma1 = []
        s_ma2 = []
        # Spatial GAT
        spatial_input = src.permute(0, 2, 3, 1).reshape(B * T, J, C)
        s_coef = self.s_coef(spatial_input)
        s_eea = self.seea(spatial_input, self.s_SPD)
        spatial_attention = []
        for i in range(len(self.spatial_scales) - 1):
            spatial_attention.append(self.spatial_gat[i](spatial_input))  # B*T，J， J
            spatial_input, s1 = self.spatial_pool[i](spatial_input, spatial_attention[i])
            s_ma1.append(s1)
        spatial_attention.append(self.spatial_gat[-1](spatial_input))
        # coef = s_coef[:, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, self.spatial_scales[0], self.spatial_scales[0])
        # sa_fusion = spatial_attention[0] * coef
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
            # coef = s_coef[:, i].unsqueeze(-1).unsqueeze(-1).repeat(1, self.spatial_scales[0], self.spatial_scales[0])
            # sa_fusion += A * coef
            sa_fusion += A
        # sa_fusion = sa_fusion / len(self.spatial_scales)
        # sa_fusion = torch.where(sa_fusion < 0.75, torch.zeros_like(sa_fusion), sa_fusion)

        # Temporal GAT
        temporal_input = src.permute(0, 3, 2, 1).reshape(B * J, T, C)
        t_coef = self.t_coef(temporal_input)
        t_eea = self.teea(temporal_input, self.t_SPD)
        temporal_attention = []
        for i in range(len(self.temporal_scales) - 1):
            temporal_attention.append(self.temporal_gat[i](temporal_input))
            temporal_input, s2 = self.temporal_pool[i](temporal_input, temporal_attention[i])
            s_ma2.append(s2)
        temporal_attention.append(self.temporal_gat[-1](temporal_input))
        # ta_fusion = temporal_attention[0] * t_coef[:, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, self.temporal_scales[0],
        #                                                                                     self.temporal_scales[0])
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
            # ta_fusion += A
            # ta_fusion += A * t_coef[:, 0].unsqueeze(-1).unsqueeze(-1).repeat(1, self.temporal_scales[0],
            #                                                                  self.temporal_scales[0])
            ta_fusion += A
            # ta_fusion = ta_fusion / len(self.temporal_scales)
        # ta_fusion = torch.where(ta_fusion < 0.75, torch.zeros_like(ta_fusion), ta_fusion)

        # sa_fusion += s_eea.squeeze()
        # ta_fusion += t_eea.squeeze()
        sa_fusion = sa_fusion.reshape(B, T, J, J)
        ta_fusion = ta_fusion.reshape(B, J, T, T)
        sa_fusion += self.sa_bias.unsqueeze(0).repeat(B, 1, 1, 1)
        ta_fusion += self.ta_bias.unsqueeze(0).repeat(B, 1, 1, 1)
        sa_fusion = self.softmax(sa_fusion)
        ta_fusion = self.softmax(ta_fusion)
        sa_fusion = self.dropout(sa_fusion)
        ta_fusion = self.dropout(ta_fusion)
        return sa_fusion, ta_fusion


# class DMS_STAttention(nn.Module):
#     def __init__(self, in_features, out_features, hidden_features, heads, spatial_scales, temporal_scales,
#                  attn_dropout=0.1):
#         super(DMS_STAttention, self).__init__()
#         self.spatial_scales = spatial_scales
#         self.temporal_scales = temporal_scales
#         self.in_features = in_features
#         self.hidden_features = hidden_features
#         self.out_features = out_features
#         self.heads = heads
#
#         self.W = nn.Parameter(torch.FloatTensor(heads, in_features, out_features))
#         nn.init.xavier_uniform_(self.W, gain=1.414)
#
#         self.as_src = []
#         self.as_dst = []
#         self.at_src = []
#         self.at_dst = []
#         for i in range(len(spatial_scales)):
#             self.as_src.append(nn.Parameter(torch.FloatTensor(heads, out_features, 1)).to('cuda:0'))
#             nn.init.xavier_uniform_(self.as_src[i], gain=1.414)
#             self.as_dst.append(nn.Parameter(torch.FloatTensor(heads, out_features, 1)).to('cuda:0'))
#             nn.init.xavier_uniform_(self.as_dst[i], gain=1.414)
#         for j in range(len(temporal_scales)):
#             self.at_src.append(nn.Parameter(torch.FloatTensor(heads, out_features, 1)).to('cuda:0'))
#             nn.init.xavier_uniform_(self.at_src[j], gain=1.414)
#             self.at_dst.append(nn.Parameter(torch.FloatTensor(heads, out_features, 1)).to('cuda:0'))
#             nn.init.xavier_uniform_(self.at_dst[j], gain=1.414)
#
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(attn_dropout)
#
#         self.spatial_pool = nn.ModuleList()
#         self.temporal_pool = nn.ModuleList()
#         self.spatial_gat = nn.ModuleList()
#         self.temporal_gat = nn.ModuleList()
#
#         self.s_coef = Mix_Coefficient(spatial_scales[0], in_features, len(spatial_scales))
#         self.t_coef = Mix_Coefficient(temporal_scales[0], in_features, len(temporal_scales))
#
#         for i in range(len(spatial_scales) - 1):
#             self.spatial_pool.append(DiffPoolingLayer(in_features, hidden_features, spatial_scales[i + 1]))
#         for i in range(len(temporal_scales) - 1):
#             self.temporal_pool.append(DiffPoolingLayer(in_features, hidden_features, temporal_scales[i + 1]))
#
#     def forward(self, src):
#         # src: B, T, J, C_in
#         # B, T, J, C_out
#         B, T, J, C = src.size()
#         # get spatial/temporal input and MoE coefficient
#         spatial_input = src.reshape(B * T, J, C).unsqueeze(1)
#         # todo:要不要把coef的输入改为不同尺度的输入
#         spatial_coef = self.s_coef(src.reshape(B * T, J, -1))  # B*T, s_scales
#         temporal_input = src.permute(0, 2, 1, 3).reshape(B * J, T, C).unsqueeze(1)
#         temporal_coef = self.t_coef(src.permute(0, 2, 1, 3).reshape(B * J, T, -1))  # B*J, t_scales
#
#         # set list of spatial/temporal pooling matrix and attention matrix
#         spatial_pooling = []
#         spatial_attention = []
#         temporal_pooling = []
#         temporal_attention = []
#
#         # todo:s作为左右乘的元素？
#         # spatial GAT
#         for i in range(len(self.spatial_scales)):
#             # print(self.as_src[i].device)
#             spatial_input_emb = torch.matmul(spatial_input, self.W)
#             attn_src = torch.matmul(spatial_input_emb, self.as_src[i])  # B*T, H, J, 1
#             attn_dst = torch.matmul(spatial_input_emb, self.as_dst[i])  # B*T, H, J, 1
#             attn = attn_src.expand(-1, -1, -1, self.spatial_scales[i]) + \
#                    attn_dst.expand(-1, -1, -1, self.spatial_scales[i]).permute(0, 1, 3, 2)
#             attn = self.leaky_relu(attn)
#             attn = self.softmax(attn)
#             attn = self.dropout(attn)
#             spatial_attention.append(attn)  # B*T，H, J， J
#             if i < len(self.spatial_scales) - 1:
#                 spatial_input, pooling = self.spatial_pool[i](spatial_input, spatial_attention[i])
#                 spatial_pooling.append(pooling)
#         s_coef = spatial_coef[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).\
#             repeat(1, self.heads, self.spatial_scales[0], self.spatial_scales[0])
#         sa_fusion = spatial_attention[0] * s_coef
#
#         for i in range(1, len(self.spatial_scales)):
#             A_prev = spatial_attention[i]  # 每个尺度的注意力矩阵
#             S = spatial_pooling[0]  # 初始化S为S(0,1)
#             if i >= 1:  # 如果i大于1，那么需要计算S(0,i)
#                 for j in range(1, i):  # 对每个池化矩阵进行循环
#                     S = torch.matmul(S, spatial_pooling[j])
#             A = torch.matmul(torch.matmul(S, A_prev), S.transpose(2, 3))
#             s_coef = spatial_coef[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1). \
#                 repeat(1, self.heads, self.spatial_scales[0], self.spatial_scales[0])
#             sa_fusion += A * s_coef
#
#         # temporal GAT
#         for i in range(len(self.temporal_scales)):
#             temporal_input_emb = torch.matmul(temporal_input, self.W)
#             attn_src = torch.matmul(temporal_input_emb, self.at_src[i])  # B*J, H, T, 1
#             attn_dst = torch.matmul(temporal_input_emb, self.at_dst[i])  # B*J, H, T, 1
#             attn = attn_src.expand(-1, -1, -1, self.temporal_scales[i]) + \
#                    attn_dst.expand(-1, -1, -1, self.temporal_scales[i]).permute(0, 1, 3, 2)
#             attn = self.leaky_relu(attn)
#             attn = self.softmax(attn)
#             attn = self.dropout(attn)
#             temporal_attention.append(attn)  # B*T，H, J， J
#             if i < len(self.temporal_scales) - 1:
#                 temporal_input, pooling = self.temporal_pool[i](temporal_input, temporal_attention[i])
#                 temporal_pooling.append(pooling)
#         t_coef = temporal_coef[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1). \
#             repeat(1, self.heads, self.temporal_scales[0], self.temporal_scales[0])
#         ta_fusion = temporal_attention[0] * t_coef
#         for i in range(1, len(self.temporal_scales)):
#             A_prev = temporal_attention[i]
#             S = temporal_pooling[0]
#             if i >= 1:
#                 for j in range(1, i):
#                     S = torch.matmul(S, temporal_pooling[j])
#             A = torch.matmul(torch.matmul(S, A_prev), S.transpose(2, 3))
#             t_coef = temporal_coef[:, i].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1). \
#                 repeat(1, self.heads, self.temporal_scales[0], self.temporal_scales[0])
#             ta_fusion += A * t_coef
#         # todo:加上物理连接？
#         src_emb = torch.matmul(src.unsqueeze(2), self.W)
#         sa_fusion = sa_fusion.reshape(B, T, self.heads, J, J)
#         ta_fusion = ta_fusion.reshape(B, J, self.heads, T, T)
#         return src_emb, sa_fusion, ta_fusion


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
