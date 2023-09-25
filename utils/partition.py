import torch
import torch.nn as nn
import numpy as np
import seaborn
import torch_geometric
from torch_geometric.nn.dense import dense_diff_pool, mincut_pool
from torch_geometric.nn import DenseGraphConv, GraphConv, GCNConv, GATConv
import matplotlib.pyplot as plt
from torch_geometric.nn import graclus
torch.set_printoptions(threshold=np.inf)


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


def rank3_trace(x):
    return torch.einsum('ijj->i', x)


def rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(x.size(0), x.size(1), x.size(1))
    return out


def mincut_loss(s, adj, out_adj):
    # MinCut regularization.
    mincut_num = rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = rank3_diag(d_flat)
    mincut_den = rank3_trace(torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    loss = -(mincut_num / mincut_den)
    loss = torch.mean(loss)
    return loss


def ortho_loss(s):
    # Orthogonality regularization.
    ss = torch.matmul(s.transpose(1, 2), s)
    k = s.size(-1)
    i_s = torch.eye(k).type_as(ss)
    loss = torch.norm(ss / torch.norm(ss, dim=(-1, -2), keepdim=True) - i_s / torch.norm(i_s), dim=(-1, -2))
    loss = torch.mean(loss)
    return loss


def fetch_assign_matrix(random, dim1, dim2, normalize=False):
    if random == 'uniform':
        m = torch.rand(dim1, dim2)
    elif random == 'normal':
        m = torch.randn(dim1, dim2)
    elif random == 'categorical':
        idxs = torch.multinomial((1.0/dim2)*torch.ones((dim1, dim2)), 1)
        m = torch.zeros(dim1, dim2)
        m[torch.arange(dim1), idxs.view(-1)] = 1.0

    if normalize:
        m = m / (m.sum(dim=1, keepdim=True) + 1e-15)
    return m


# class GCNLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, bias=False):
#         super().__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.lin = nn.Linear(in_channels, out_channels, bias=False)
#
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.lin.reset_parameters()
#         if self.bias is not None:
#             self.bias.data.fill_(0.0)
#
#     def forward(self, x, adj, mask=None):
#         B, N, C = adj.size()
#         out = self.lin(x).detach()
#
#         if mask is not None:
#             adj = adj * mask.view(B, N, N).to(x.dtype)
#         # else:
#         #     if N == 22:
#         #         I22_link = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
#         #         J22_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
#         #         mask = cal_spatial_adj(I22_link, J22_link, N)
#         #         mask = torch.from_numpy(mask).to('cuda:0')
#         #         adj = adj * mask.unsqueeze(0).to(x.dtype)
#         #     elif N == 10:
#         #         mask = cal_temporal_adj(N)
#         #         mask = torch.from_numpy(mask).to('cuda:0')
#         #         adj = adj * mask.unsqueeze(0).to(x.dtype)
#
#         out = torch.matmul(adj, out)
#
#         if self.bias is not None:
#             out = out + self.bias
#         return out
#
#
# class PoolingGNN(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features, normalize=False, linear=True):
#         super(PoolingGNN, self).__init__()
#         # self.conv1 = GCNLayer(in_features, hidden_features, normalize)
#         # self.bn1 = nn.BatchNorm1d(hidden_features)
#         # self.conv2 = GCNLayer(hidden_features, hidden_features, normalize)
#         # self.bn2 = nn.BatchNorm1d(hidden_features)
#         # self.conv3 = GCNLayer(hidden_features, out_features, normalize)
#         self.bn3 = nn.BatchNorm1d(out_features)
#         self.conv_1 = GCNLayer(in_features, out_features, normalize)
#         # self.bn_1 = nn.BatchNorm1d(out_features)
#
#         # if linear is True:
#         #     self.linear = nn.Linear(hidden_features + out_features, out_features)
#         # else:
#         #     self.linear = None
#         #
#         # if in_features != out_features:
#         #     self.residual = nn.Sequential(nn.Linear(in_features, out_features))
#         # else:
#         #     self.residual = nn.Identity()
#
#     def bn(self, i, x):
#         batch_size, num_nodes, num_features = x.size()
#         x = x.view(-1, num_features)
#         x = getattr(self, f'bn{i}')(x)
#         x = x.view(batch_size, num_nodes, num_features)
#         return x
#
#     def forward(self, x, adj, mask=None):
#         batch_size, num_nodes, in_channels = x.size()
#
#         # x0 = self.residual(x)
#         # x1 = self.bn(1, self.conv1(x, adj, mask).relu())
#         # x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
#         # x3 = self.bn(3, self.conv3(x2, adj, mask).relu())
#         # x3 = x3 + x0
#
#         x3 = self.bn(3, self.conv_1(x, adj, mask).relu())
#         # x = torch.cat([x1, x2, x3], dim=-1)
#
#         # if self.linear is not None:
#         #     x = self.linear(x).relu()
#         return x3


# class PoolingLayer(nn.Module):
#     def __init__(self, in_features, hidden_features, num_clusters):
#         super(PoolingLayer, self).__init__()
#         self.in_features = in_features
#         self.num_clusters = num_clusters
#         self.GNN_Pool = PoolingGNN(in_features, 64, num_clusters)
#         # todo:embedding?
#         # self.GNN_Embed = PoolingGNN(in_features, hidden_features, in_features, linear=False)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, src, adj, adj_sparse=None, mask=None):
#         # src: B, N, C
#         # adj: B, N, N
#         B, N, _ = adj.size()
#         S = self.GNN_Pool(src, adj, mask)
#         # todo: 要不要除一个系数
#         S = self.softmax(S)  # B, N, num_cluster
#         # seaborn.heatmap(data=S[0].cpu().detach())
#         # plt.show()
#
#         # 创建一个与矩阵形状相同的全零矩阵，然后根据最大值的索引将对应位置设为1
#         #TODO:
#         # max_indices = torch.argmax(S, dim=-1)
#         # one_hot_S = torch.zeros_like(S)
#         # one_hot_S.scatter_(-1, max_indices.unsqueeze(-1), 1)
#
#         one_hot_S = S.detach()
#         # S_degree = (torch.sum(S, dim=1)+0.0001).unsqueeze(1).repeat(1, N, 1)  # B, H, N, num_cluster
#         # # todo: src embed->return?
#         # src = self.GNN_Embed(src, adj, mask)
#         # out = torch.matmul(torch.div(S, S_degree).transpose(1, 2), src)  # B, H, num_cluster, N & B, H, N, C
#         out = torch.matmul(one_hot_S.transpose(1, 2), src)
#
#         if adj_sparse is None:
#             adj_sparse = torch.ones_like(adj)
#
#         adj_new = torch.matmul(torch.matmul(one_hot_S.transpose(1, 2), adj_sparse), one_hot_S)
#
#         for i in range(self.num_clusters):
#             adj_new[:, i, i] = 0
#
#         adj_new[adj_new > 0] = 1
#
#         out_adj = torch.matmul(one_hot_S.transpose(1, 2), adj)
#         out_adj = torch.matmul(out_adj, one_hot_S)
#         # seaborn.heatmap(data=adj[0].detach().cpu(), square=True)
#         # plt.show()
#         m_loss = mincut_loss(one_hot_S, adj, out_adj)
#         o_loss = ortho_loss(one_hot_S)
#
#         return out, one_hot_S, adj_new, m_loss, o_loss  # [B, H, num_cluster, C] [B, H, N, num_cluster]

# class PoolingLayer(nn.Module):
#     def __init__(self, in_features, hidden_features, num_clusters, num_node):
#         super(PoolingLayer, self).__init__()
#         self.in_features = in_features
#         self.num_clusters = num_clusters
#         self.num_node = num_node
#         self.emb = nn.Linear(in_features, hidden_features)
#         # self.conv = DenseGraphConv(in_features, hidden_features, aggr='add')
#         self.pools = nn.Linear(hidden_features, num_clusters, bias=False)
#         self.ReLU = nn.ReLU()
#         # self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, src, adj, adj_sparse=None, mask=None):
#         # src: B, N, C
#         # adj: B, N, N
#         B, N, _ = adj.size()
#         src_emb = self.ReLU(self.emb(src))
#         S = self.pools(src_emb)
#         # S_ = self.softmax(S)  # B, N, num_cluster
#
#         # 创建一个与矩阵形状相同的全零矩阵，然后根据最大值的索引将对应位置设为1
#         max_indices = torch.argmax(S, dim=-1)
#         one_hot_S = torch.zeros_like(S)
#         one_hot_S.scatter_(-1, max_indices.unsqueeze(-1), 1)
#         one_hot_D = torch.sum(one_hot_S, dim=1).unsqueeze(1).repeat(1, N, 1)
#         one_hot_S = one_hot_S / (one_hot_D + 1e-15)
#
#         # Todo:
#         # one_hot_S = S.detach()
#         # one_hot_D = torch.sum(one_hot_S, dim=1).unsqueeze(1).repeat(1, N, 1)
#         # one_hot_S = one_hot_S / (one_hot_D + 1e-15)
#
#         if adj_sparse is None:
#             adj_sparse = torch.ones_like(adj)
#
#         adj_sparse_new = torch.matmul(torch.matmul(one_hot_S.transpose(1, 2), adj_sparse), one_hot_S)
#
#         for i in range(self.num_clusters):
#             adj_sparse_new[:, i, i] = 0
#         adj_sparse_new[adj_sparse_new > 0] = 1
#         # seaborn.heatmap(data=adj[0].detach().cpu(), square=True)
#         # plt.show()
#         out, _, m_loss, o_loss = mincut_pool.dense_mincut_pool(src, adj, one_hot_S)
#
#         return out, one_hot_S, adj_sparse_new, m_loss, o_loss


class PoolingLayer(nn.Module):
    def __init__(self, in_features, hidden_features, num_clusters, num_node):
        super(PoolingLayer, self).__init__()
        self.in_features = in_features
        self.num_clusters = num_clusters
        self.num_node = num_node
        self.emb = nn.Linear(in_features, hidden_features)
        # self.conv = DenseGraphConv(in_features, hidden_features, aggr='add')
        self.pools = nn.Linear(hidden_features, num_clusters, bias=False)
        self.ReLU = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, adj, adj_sparse=None, mask=None):
        # src: B, N, C
        # adj: B, N, N
        B, N, _ = adj.size()
        src_emb = self.ReLU(self.emb(src))
        S = self.pools(src_emb)
        S = self.softmax(S)  # B, N, num_cluster

        # 创建一个与矩阵形状相同的全零矩阵，然后根据最大值的索引将对应位置设为1
        # max_indices = torch.argmax(S, dim=-1)
        # one_hot_S = torch.zeros_like(S)
        # one_hot_S.scatter_(-1, max_indices.unsqueeze(-1), 1)
        # one_hot_D = torch.sum(one_hot_S, dim=1).unsqueeze(1).repeat(1, N, 1)
        # one_hot_S = one_hot_S / (one_hot_D + 1e-15)

        # Todo:
        one_hot_S = S.detach()
        one_hot_D = torch.sum(one_hot_S, dim=1).unsqueeze(1).repeat(1, N, 1)
        one_hot_S = one_hot_S / (one_hot_D + 1e-15)

        if adj_sparse is None:
            adj_sparse = torch.ones_like(adj)

        adj_sparse_new = torch.matmul(torch.matmul(one_hot_S.transpose(1, 2), adj_sparse), one_hot_S)

        for i in range(self.num_clusters):
            adj_sparse_new[:, i, i] = 0
        adj_sparse_new[adj_sparse_new > 0] = 1
        # seaborn.heatmap(data=adj[0].detach().cpu(), square=True)
        # plt.show()
        out, _, m_loss, o_loss = mincut_pool.dense_mincut_pool(src, adj, one_hot_S)

        return out, one_hot_S, adj_sparse_new, m_loss, o_loss

if __name__ == '__main__':
    pass
