import torch
import torch.nn as nn
from torch_geometric.nn.dense import dense_diff_pool


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias = True):
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
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, C = adj.size()

        out = self.lin(x)
        out = torch.matmul(adj, out)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out


class PoolingGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, normalize=False, linear=True) :
        super(PoolingGNN, self).__init__()
        self.conv1 = GCNLayer(in_features, hidden_features, normalize)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.conv2 = GCNLayer(hidden_features, hidden_features, normalize)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.conv3 = GCNLayer(hidden_features, out_features, normalize)
        self.bn3 = nn.BatchNorm1d(out_features)

        if linear is True:
            self.linear = nn.Linear(2 * hidden_features + out_features, out_features)
        else:
            self.linear = None

    def bn(self, i, x):
        batch_size, num_nodes, num_features = x.size()
        x = x.view(-1, num_features)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_features)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.linear is not None:
            x = self.linear(x).relu()
        return x


class DiffPoolingLayer(nn.Module):
    def __init__(self, in_features, hidden_features, num_clusters):
        super(DiffPoolingLayer, self).__init__()
        self.in_features = in_features
        self.num_clusters = num_clusters
        self.GNN_Pool = PoolingGNN(in_features, hidden_features, num_clusters)
        # todo:embedding?
        self.GNN_Embed = PoolingGNN(in_features, hidden_features, hidden_features, linear=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, adj, mask=None, device='cuda:0'):
        # adj = torch.mean(adj, dim=1)
        B, H, N, _ = adj.size()
        C = src.size()[2]
        output = torch.zeros([B, H, self.num_clusters, C]).to(device)
        s = torch.zeros([B, H, N, self.num_clusters]).to(device)
        l = 0
        e = 0
        for i in range(H):
            s[:, i, :, :] = self.GNN_Pool(src, adj[:, i, :, :], mask)
            # todo: src embed->return?
            # src = self.GNN_Embed(src, adj, mask)
            s = self.softmax(s)
            output[:, i, :, :], _, l1, e1 = dense_diff_pool(src, adj[:, i, :, :], s[:, i, :, :], mask)
            l += l1
            e += e1
        output = torch.mean(output, dim=1)
        return output, s, l, e  # [B, N, C] [B, N, C, Cluster]


if __name__ == '__main__':
    pass
