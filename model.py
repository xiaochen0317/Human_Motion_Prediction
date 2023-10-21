import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.DCT import get_dct_matrix


# Parameter Initialization
def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PositionalEmbedding(nn.Module):
    def __init__(self, J, T, d_model, dropout=0.1):
        super().__init__()
        self.J = J
        self.T = T
        self.joint_PE = nn.Parameter(torch.zeros(J, d_model))
        self.time_PE = nn.Parameter(torch.zeros(T, d_model))
        self.dropout = nn.Dropout(dropout)
        torch.nn.init.normal_(self.joint_PE, std=0.02)
        torch.nn.init.normal_(self.time_PE, std=0.02)

    def forward_spatial(self):
        p_joint = self.joint_PE.unsqueeze(0).repeat(self.T, 1, 1)
        return self.dropout(p_joint)

    def forward_temporal(self):
        p_time = self.time_PE.unsqueeze(1).repeat(1, self.J, 1)
        return self.dropout(p_time)

    def forward(self):
        p_joint = self.forward_spatial()
        p_time = self.forward_temporal()
        p = p_joint + p_time
        return self.dropout(p)

    def forward_spatial_relation(self):
        p = self.joint_PE
        p_i = p.unsqueeze(-2)
        p_j = p.unsqueeze(-3)
        p = p_i + p_j
        return self.dropout(p)

    def forward_temporal_relation(self):
        p = self.time_PE
        p_i = p.unsqueeze(-2)
        p_j = p.unsqueeze(-3)
        p = p_i + p_j
        return self.dropout(p)


# Generalized MLP (Dropout form)
# class MLP(nn.Module):
#     def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), act_layer=None, dropout=-1):
#         super().__init__()
#         dims = (in_feat,) + hid_feat + (out_feat,)
#
#         self.layers = nn.ModuleList()
#         for i in range(len(dims) - 1):
#             self.layers.append(nn.Linear(dims[i], dims[i + 1]))
#
#         self.activation = act_layer if act_layer is not None else lambda x: x
#         self.drop = nn.Dropout(dropout) if dropout != -1 else lambda x: x
#
#     def forward(self, x):
#         x = self.layers[0](x)
#         for i in range(1, len(self.layers)):
#             x = self.activation(x)
#             x = self.drop(x)
#             x = self.layers[i](x)
#         return x


# Generalized MLP (BatchNorm form)
class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat=(1024, 512), act_layer=None, dropout=-1):
        super().__init__()
        dims = (in_feat,) + hid_feat + (out_feat,)

        self.layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            self.bn.append(nn.BatchNorm1d(dims[i + 1]))

        self.activation = act_layer if act_layer is not None else lambda x: x

    def forward(self, x):
        x = self.layers[0](x)
        x = self.bn[0](x)
        for i in range(1, len(self.layers)):
            x = self.activation(x)
            x = self.layers[i](x)
            x = self.bn[i](x)
        return x


# 2-layer MLP (Dropout form)
# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


# 2-layer MLP (BatchNorm form)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm1 = nn.BatchNorm1d(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.norm2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale, attn_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, bias=None):
        # q x k^T
        attn = torch.matmul(q / self.scale, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        if bias is not None:
            attn = attn + bias
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)
        self.W_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.W_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = ScaledDotProductAttention(self.scale, attn_drop)

        self.R_conv = nn.Linear(dim, num_heads, bias=qkv_bias)
        self.R_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, relation_feature, mask=None):
        # q: B, N1, C
        # k, v : B, N2, C
        # relation_feature: B, N1, N2, C
        B, N1, C = q.size()
        N2 = k.size()[1]
        H = self.num_heads
        HS = C // self.num_heads
        q = self.W_q(q).view(B, N1, H, HS).permute(0, 2, 1, 3)  # [B, H, N1, C//H]
        k = self.W_k(k).view(B, N2, H, HS).permute(0, 2, 1, 3)  # [B, H, N2, C//H]
        v = self.W_v(v).view(B, N2, H, HS).permute(0, 2, 1, 3)  # [B, H, N2, C//H]
        R_qkv = self.R_qk(relation_feature).view(B, N1, N2, 2, H, HS).permute(3, 0, 4, 1, 2, 5)
        # [2, B, H, N1, N2, C//H]
        R_q, R_k = R_qkv[0], R_qkv[1]

        attn_J = (q @ k.transpose(-2, -1))  # [B, H, N1, N2]
        attn_R_linear = self.R_conv(relation_feature).reshape(B, N1, N2, H).permute(0, 3, 1, 2)  # [B, H, N1, N2]
        attn_R_qurt = (R_q.unsqueeze(-2) @ R_k.unsqueeze(-1)).squeeze()  # [B, H, N1, N2]

        attn = (attn_J + attn_R_linear + attn_R_qurt) * self.scale  # [B, H, N1, N2]
        # attn = attn_J * self.scale  # [B, H, N1, N2]
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, H, N1, N2]

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)  # [B, N1, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class SpatialAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, spatial_scales, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.spatial_scales = spatial_scales

        self.norm_attn1 = nn.LayerNorm(dim)
        self.norm_attn2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pooling_layer = nn.ModuleList()
        for i in range(len(spatial_scales) - 1):
            # todo: pooling_layer加上输入函数的GCN?
            self.pooling_layer.append(nn.Sequential(Mlp(dim, 2 * dim, self.spatial_scales[i + 1]), nn.Softmax(-1)))

        self.attention = nn.ModuleList()
        for i in range(len(spatial_scales)):
            self.attention.append(Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop))

        self.spatial_relation_emb = MLP(2, dim, (dim,), act_layer=nn.GELU())
        self.ms_fusion = Mlp(dim * len(spatial_scales), dim, dim)

    def forward(self, src, spatial_relation, mask=None):
        # src: B, T, J, C
        # spatial_relation : B, T, J, J, 2
        B, T, J, C = src.size()
        src = src.view(B * T, J, C)
        spatial_relation = self.spatial_relation_emb(spatial_relation).view(B * T, J, J, C)

        attn_list = []
        out_list = None
        joint_feature_g = src
        joint_feature_l = src
        relation_feature = spatial_relation

        for i in range(len(self.spatial_scales)):
            q, attn = self.drop_path(
                self.attention[i](self.norm_attn1(joint_feature_g), self.norm_attn1(joint_feature_l),
                                  self.norm_attn1(joint_feature_l), self.norm_attn2(relation_feature), mask))
            q = joint_feature_g + q  # residual connection
            out_list = torch.concat((out_list, q.unsqueeze(-1)), dim=-1) if out_list is not None else q  # B*T, J, C*scl
            attn_list.append(attn)
            if i < len(self.spatial_scales) - 1:
                S = self.pooling_layer[i](joint_feature_l)
                # todo: S进行归一化？
                joint_feature_l = S.transpose(1, 2) @ joint_feature_l
                relation_feature = S.transpose(1, 2) @ relation_feature
        output = self.ms_fusion(out_list)
        return output


class TemporalAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads, temporal_scales, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temporal_scales = temporal_scales

        self.norm_attn1 = nn.LayerNorm(dim)
        self.norm_attn2 = nn.LayerNorm(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.pooling_layer = nn.ModuleList()
        for i in range(len(temporal_scales) - 1):
            # todo: pooling_layer加上输入函数的GCN?
            self.pooling_layer.append(nn.Sequential(Mlp(dim, 2 * dim, self.temporal_scales[i + 1]), nn.Softmax(-1)))

        self.attention = nn.ModuleList()
        for i in range(len(temporal_scales)):
            self.attention.append(Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop))

        self.temporal_relation_emb = MLP(2, dim, (dim,), act_layer=nn.GELU())
        self.ms_fusion = Mlp(dim * len(temporal_scales), dim, dim)

    def forward(self, src, temporal_relation, mask=None):
        # src: B, T, J, C
        # spatial_relation : B, J, T, T, C
        B, T, J, C = src.size()
        src = src.permute(0, 2, 1, 3).contiguous().view(B * J, T, C)
        temporal_relation = self.temporal_relation_emb(temporal_relation).view(B * J, T, T, C)

        attn_list = []
        out_list = None
        joint_feature_g = src
        joint_feature_l = src
        relation_feature = temporal_relation

        for i in range(len(self.temporal_scales)):
            q, attn = self.drop_path(
                self.attention[i](self.norm_attn1(joint_feature_g), self.norm_attn1(joint_feature_l),
                                  self.norm_attn1(joint_feature_l), self.norm_attn2(relation_feature), mask))
            q = joint_feature_g + q
            out_list = torch.concat((out_list, q.unsqueeze(-1)), dim=-1) if out_list is not None else q
            attn_list.append(attn)
            if i < len(self.temporal_scales) - 1:
                S = self.pooling_layer[i](joint_feature_g)
                # todo: S进行归一化？
                joint_feature_l = S.transpose(1, 2) @ joint_feature_g
                relation_feature = S.transpose(1, 2) @ relation_feature
        output = self.ms_fusion(out_list)
        return output


class ST_Block(nn.Module):
    def __init__(self, dim, num_heads, spatial_scales, temporal_scales, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales

        self.spatial_attention = SpatialAttentionLayer(dim, num_heads, spatial_scales, qkv_bias, qk_scale, drop,
                                                       attn_drop, drop_path)
        self.temporal_attention = TemporalAttentionLayer(dim, num_heads, temporal_scales, qkv_bias, qk_scale, drop,
                                                         attn_drop, drop_path)
        self.spatial_norm = nn.BatchNorm2d(dim)
        self.temporal_norm = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()

    def forward(self, src, spatial_relation, temporal_relation, spatial_mask=None, temporal_mask=None):
        # src: B, T, J, C
        # todo: dropout?
        B, T, J, C = src.size()
        output = self.spatial_attention(src, spatial_relation, spatial_mask).view(B, T, J, C).permute(0, 3, 1, 2) \
            .contiguous()
        output = self.gelu(self.spatial_norm(output)).permute(0, 2, 3, 1).contiguous()
        output = self.temporal_attention(output, temporal_relation, temporal_mask).view(B, J, T, C).permute(0, 3, 2, 1) \
            .contiguous()
        output = self.temporal_norm(output).permute(0, 2, 3, 1).contiguous()
        output = output + src
        return output


class TS_Block(nn.Module):
    def __init__(self, dim, num_heads, spatial_scales, temporal_scales, qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales

        self.spatial_attention = SpatialAttentionLayer(dim, num_heads, spatial_scales, qkv_bias, qk_scale, drop,
                                                       attn_drop, drop_path)
        self.temporal_attention = TemporalAttentionLayer(dim, num_heads, temporal_scales, qkv_bias, qk_scale, drop,
                                                         attn_drop, drop_path)
        self.spatial_norm = nn.BatchNorm2d(dim)
        self.temporal_norm = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()

    def forward(self, src, spatial_relation, temporal_relation, spatial_mask=None, temporal_mask=None):
        # src: B, T, J, C
        # todo: dropout?
        B, T, J, C = src.size()
        output = self.temporal_attention(src, temporal_relation, temporal_mask).view(B, J, T, C).permute(0, 3, 2, 1) \
            .contiguous()
        output = self.temporal_norm(output).permute(0, 2, 3, 1).contiguous()
        output = self.spatial_attention(output, spatial_relation, spatial_mask).view(B, T, J, C).permute(0, 3, 1, 2) \
            .contiguous()
        output = self.gelu(self.spatial_norm(output)).permute(0, 2, 3, 1).contiguous()
        output = output + src
        return output


class TCN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout, bias=True):
        super(TCN_Layer, self).__init__()
        self.kernel_size = kernel_size
        padding = (
            (kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1

        self.block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                      nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True)]

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        output = self.block(x)
        return output


class Model(nn.Module):
    def __init__(self, d_in=3, d_out=3, dim=128, num_heads=8, spatial_scales=[22,12,5], temporal_scales=[10,6,3],
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., tcn_dropout=0., drop_path=0.,
                 input_time_frame=10, output_time_frame=25, joints_to_consider=22, dct_n=10,
                 num_encoder_layer=4, num_decoder_layers=4):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dim = dim
        self.num_heads = num_heads
        self.spatial_scales = spatial_scales
        self.temporal_scales = temporal_scales
        self.num_encoder_layer = num_encoder_layer
        self.num_decoder_layers = num_decoder_layers
        self.dct_n = dct_n
        self.input_time_frame = input_time_frame
        self.output_time_frame = output_time_frame

        # todo: 考虑输入数据时域扩维
        self.DCT, _ = get_dct_matrix(input_time_frame)
        _, self.IDCT = get_dct_matrix(output_time_frame)

        self.joint_emb = MLP(d_in, dim, (128,), act_layer=nn.GELU())
        self.PE = PositionalEmbedding(joints_to_consider, input_time_frame, dim, dropout=drop)
        self.norm_layer = nn.LayerNorm(dim)

        self.Encoder = nn.ModuleList([
            ST_Block(dim, num_heads, spatial_scales, temporal_scales, qkv_bias, qk_scale, drop, attn_drop, drop_path)
            for _ in range(num_encoder_layer)])

        self.Decoder = nn.ModuleList()
        self.Decoder.append(TCN_Layer(input_time_frame, output_time_frame, [3, 3], tcn_dropout))
        for i in range(1, num_decoder_layers):
            self.Decoder.append(TCN_Layer(output_time_frame, output_time_frame, [3, 3], tcn_dropout))
        self.Decoder_channel = MLP(dim, d_out, (64,), act_layer=nn.GELU())
        self.prelus = nn.ModuleList()
        for j in range(num_decoder_layers):
            self.prelus.append(nn.PReLU())

        # todo: 加全局residual?
        # self.residual = nn.Sequential(nn.Conv2d(input_time_frame, output_time_frame, kernel_size=1, stride=(1, 1)),
        #                               nn.BatchNorm2d(output_time_frame))
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.normal_(m.weight, 1.0, 0.02)

    def cal_spatial_relation(self, src, spatial_connection):
        # src: B, T, J, C
        # spatial_connection: J, J
        dist = torch.cdist(src, src, p=2)
        connection = spatial_connection.unsqueeze(0).unsqueeze(0)
        spatial_relation = torch.cat((dist.unsqueeze(-1), connection.unsqueeze(-1)), dim=-1)
        return spatial_relation

    def cal_temporal_relation(self, src, temporal_connection):
        # src: B, T, J, C
        # spatial_connection: T, T
        dist = torch.cdist(src, src, p=2)
        connection = temporal_connection.unsqueeze(0).unsqueeze(0)
        temporal_relation = torch.cat((dist.unsqueeze(-1), connection.unsqueeze(-1)), dim=-1)
        return temporal_relation
    # def spatial_relation_update(self, x_joint, spatial_relation):
    #     # x_joint: B, T, J, C_hid
    #     # spatial_relation: B, T, J, J, C_hid
    #     B, T ,J, _ = x_joint.shape
    #     x_joint_i = x_joint.unsqueeze(2).repeat(1, 1, J, 1, 1)
    #     x_joint_j = x_joint.unsqueeze(3).repeat(1, 1, 1, J, 1)
    #     spatial_relation_rev = torch.swapaxes(spatial_relation, 2, 3)
    #     relation_concat = torch.cat((spatial_relation, spatial_relation_rev, x_joint_i, x_joint_j), dim=-1)

    def forward(self, x_joint, x_spatial_connection, x_temporal_connection, spatial_mask=None, temporal_mask=None):
        # x_joint: B, T, J, C
        # x_spatial_connection: J, J
        # x_temporal_connection: T, T
        v_joint = x_joint[:, 1:] - x_joint[:, :-1]
        v_joint = torch.cat([v_joint, v_joint[:, -1].unsqueeze(1)], dim=1)
        a_joint = v_joint[:, 1:] - v_joint[:, :-1]
        a_joint = torch.cat([a_joint, a_joint[:, -1].unsqueeze(1)], dim=1)
        x_joint = torch.cat([x_joint, v_joint, a_joint], dim=-1)

        x_joint_emb = self.joint_emb(x_joint)  # B, T, J, C_hid
        x_spatial_relation = self.cal_spatial_relation(x_joint, x_spatial_connection)
        x_temporal_relation = self.cal_temporal_relation(x_joint, x_temporal_connection)
        x_spatial_relation_emb = x_spatial_relation
        x_temporal_relation_emb = x_temporal_relation
        pe_joint = self.PE.forward()
        # pe_spatial_relation = self.PE.forward_spatial_relation()
        # pe_temporal_relation = self.PE.forward_temporal_relation()

        for i in range(len(self.Encoder)):
            # todo: 每个encoder层都加PE吗？
            x_joint_emb = x_joint_emb + pe_joint
            # todo: relation是否要加PE?
            # x_spatial_relation_emb = x_spatial_relation_emb + pe_spatial_relation
            # x_temporal_relation_emb = x_temporal_relation_emb + pe_temporal_relation
            x_joint_emb = self.Encoder[i](x_joint_emb, x_spatial_relation_emb, x_temporal_relation_emb, spatial_mask,
                                          temporal_mask)
            x_spatial_relation_emb = self.cal_spatial_relation(x_joint_emb, x_spatial_connection)
            x_temporal_relation_emb = self.cal_temporal_relation(x_joint_emb, x_temporal_connection)

        x_joint = self.prelus[0](self.Decoder[0](x_joint))
        for i in range(1, self.num_decoder_layers):
            x_joint = self.prelus[i](self.Decoder[i](x_joint)) + x_joint  # residual connection
        x_joint = self.Decoder_channel(x_joint_emb)
        return x_joint
