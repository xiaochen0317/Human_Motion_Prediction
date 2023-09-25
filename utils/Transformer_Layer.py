import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
        return encoder_output


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


class Positional_Encoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(Positional_Encoding, self).__init__()
        self.register_buffer('pos_table1', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table2', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table3', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_pisition, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_pisition)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    # todo  单人预测怎么改？
    def forward(self, x, n_person):
        p = self.pos_table1[:, :x.size(1)].clone().detach()
        return x + p

    def forward2(self, x, n_person):
        p = self.pos_table2[:, :int(x.shape[1] / n_person)].clone().detach()
        p = p.repeat(1, n_person, 1)
        return x + p
