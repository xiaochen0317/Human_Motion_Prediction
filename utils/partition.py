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
        self.bias.data.fill_(0.0)

    def forward(self, x, adj, mask= None):
        """"
        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
                The adjacency tensor is broadcastable in the batch dimension,
                resulting in a shared adjacency matrix for the complete batch.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            self_connected (bool, optional): If set to :obj:`False`, the layer will
                not automatically add self-loops to the adjacency matrices.
                (default: :obj:`True`)
        """
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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src, adj, mask=None):
        s = self.GNN_Pool(src, adj, mask)
        output, adj, l1, e1 = dense_diff_pool(src, adj, s, mask)
        return output, l1, e1


if __name__ == '__main__':
    pass