import torch
import torch.nn as nn
from torch_geometric.nn.dense import dense_diff_pool


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

    def forward(self, x, adj, mask= None):
        B, N, C = adj.size()
        out = self.lin(x).detach()
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias
        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

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


class DiffPoolingLayer(nn.Module):
    def __init__(self, in_features, hidden_features, num_clusters):
        super(DiffPoolingLayer, self).__init__()
        self.in_features = in_features
        self.num_clusters = num_clusters
        self.GNN_Pool = PoolingGNN(in_features, hidden_features, num_clusters)
        # todo:embedding?
        # self.GNN_Embed = PoolingGNN(in_features, hidden_features, hidden_features, linear=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, adj, mask=None):
        # src: B, H, N, C
        # adj: B, H, N, N
        B, N, _ = adj.size()
        # src = src.unsqueeze(1).repeat(1, H, 1, 1) if src.dim() == 3 else src
        # S = torch.zeros([B, H, N, self.num_clusters]).to('cuda:0')
        S = self.GNN_Pool(src, adj, mask)
        # for i in range(H):
            # S[:, i, :, :] = self.GNN_Pool(src[:, i, :, :], adj[:, i, :, :], mask)
        # todo: 要不要除一个系数
        # S = self.softmax(S)  # B, H, N, num_cluster
        # print(S)
        # S = nn.functional.gumbel_softmax(S, tau=0.1)
        S = nn.functional.gumbel_softmax(S, tau=0.1, hard=True)

        S_degree = (torch.sum(S, dim=1)+0.0001).unsqueeze(1).repeat(1, N, 1)  # B, H, N, num_cluster
        # todo: src embed->return?
        # src = self.GNN_Embed(src, adj, mask)
        out = torch.matmul(torch.div(S, S_degree).transpose(1, 2), src)  # B, H, num_cluster, N & B, H, N, C
        return out, S  # [B, H, num_cluster, C] [B, H, N, num_cluster]


if __name__ == '__main__':
    pass
