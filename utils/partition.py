import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn.dense import dense_diff_pool, mincut_pool


def cal_spatial_adj(I_link, J_link, n_nodes):
    A = np.zeros([n_nodes, n_nodes])
    for i in range(len(I_link)):
        A[I_link[i], J_link[i]] = 1
        A[J_link[i], I_link[i]] = 1
    for j in range(n_nodes):
        A[j, j] = 1
    return A


def cal_temporal_adj(frames):
    A = np.zeros([frames, frames])
    for i in range(frames - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
        A[i, i] = 1
    return A


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj, mask=None):
        B, N, C = adj.size()
        out = self.lin(x).detach()

        if mask is not None:
            adj = adj * mask.view(B, N, N).to(x.dtype)
        # else:
        #     if N == 22:
        #         I22_link = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
        #         J22_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        #         mask = cal_spatial_adj(I22_link, J22_link, N)
        #         mask = torch.from_numpy(mask).to('cuda:0')
        #         adj = adj * mask.unsqueeze(0).to(x.dtype)
        #     elif N == 10:
        #         mask = cal_temporal_adj(N)
        #         mask = torch.from_numpy(mask).to('cuda:0')
        #         adj = adj * mask.unsqueeze(0).to(x.dtype)

        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias
        return out


class PoolingGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, normalize=False, linear=True):
        super(PoolingGNN, self).__init__()
        self.conv1 = GCNLayer(in_features, hidden_features, normalize)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.conv2 = GCNLayer(hidden_features, hidden_features, normalize)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.conv3 = GCNLayer(hidden_features, out_features, normalize)
        self.bn3 = nn.BatchNorm1d(out_features)
        # self.conv_1 = GCNLayer(in_features, out_features, normalize)
        # self.bn_1 = nn.BatchNorm1d(out_features)

        if linear is True:
            self.linear = nn.Linear(hidden_features + out_features, out_features)
        else:
            self.linear = None

        if in_features != out_features:
            self.residual = nn.Sequential(nn.Linear(in_features, out_features))
        else:
            self.residual = nn.Identity()

    def bn(self, i, x):
        batch_size, num_nodes, num_features = x.size()
        x = x.view(-1, num_features)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_features)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = self.residual(x)
        x1 = self.bn(1, self.conv1(x, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())
        x3 = x3 + x0

        # x = torch.cat([x1, x2, x3], dim=-1)

        # if self.linear is not None:
        #     x = self.linear(x).relu()
        return x3


class PoolingLayer(nn.Module):
    def __init__(self, in_features, hidden_features, num_clusters):
        super(PoolingLayer, self).__init__()
        self.in_features = in_features
        self.num_clusters = num_clusters
        self.GNN_Pool = PoolingGNN(in_features, 32, num_clusters)
        # todo:embedding?
        # self.GNN_Embed = PoolingGNN(in_features, hidden_features, hidden_features, linear=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, adj, mask=None):
        # src: B, N, C
        # adj: B, N, N
        B, N, _ = adj.size()
        S = self.GNN_Pool(src, adj, mask)
        # todo: 要不要除一个系数
        S = self.softmax(S)  # B, H, N, num_cluster

        S_degree = (torch.sum(S, dim=1)+0.0001).unsqueeze(1).repeat(1, N, 1)  # B, H, N, num_cluster
        # todo: src embed->return?
        # src = self.GNN_Embed(src, adj, mask)
        out = torch.matmul(torch.div(S, S_degree).transpose(1, 2), src)  # B, H, num_cluster, N & B, H, N, C
        return out, S  # [B, H, num_cluster, C] [B, H, N, num_cluster]


if __name__ == '__main__':
    pass
