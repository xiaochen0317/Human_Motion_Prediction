import torch
import math


# DCT Operation
def get_dct_matrix(N):
    dct_matrix = torch.eye(N)
    for k in range(N):
        for i in range(N):
            w = math.sqrt(2 / N)
            if k == 0:
                w = math.sqrt(1 / N)
            dct_matrix[k, i] = w * math.cos(math.pi * (i + 1 / 2) * k / N)
    idct_matrix = torch.inverse(dct_matrix)
    return dct_matrix, idct_matrix


def dct_transform_torch(data, dct_matrix, dct_n):
    assert data.dim() == 3  # data should have 3 dimensions (batch_size, features, seq_len)
    dct_matrix = dct_matrix.float().cuda(data.device)
    batch_size, features, seq_len = data.shape
    data = data.contiguous().view(-1, seq_len)
    data = data.permute(1, 0)
    out_data = torch.matmul(dct_matrix[:dct_n, :], data)
    out_data = out_data.permute(1, 0).contiguous().view(-1, features, dct_n)
    return out_data


def reverse_dct_torch(dct_data, idct_matrix, seq_len):
    assert dct_data.dim() == 3  # dct_data should have 3 dimensions (batch_size, features, dct_n)
    idct_matrix = idct_matrix.float().cuda(dct_data.device)
    batch_size, features, dct_n = dct_data.shape
    dct_data = dct_data.permute(2, 0, 1).contiguous().view(dct_n, -1)
    out_data = torch.matmul(idct_matrix[:, :dct_n], dct_data)
    out_data = out_data.contiguous().view(seq_len, batch_size, -1).permute(1, 2, 0)
    return out_data