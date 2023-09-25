# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class GraphConvolution(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(GraphConvolution, self).__init__()
#         self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#
#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         return output
#
#
# class SpectralClustering(nn.Module):
#     def __init__(self, n_clusters, adj, bias=False):
#         super(SpectralClustering, self).__init__()
#         self.gc = GraphConvolution(adj.size(1), n_clusters)
#         self.adj = adj
#         self.bias = bias
#         if self.bias:
#             self.b = nn.Parameter(torch.FloatTensor(n_clusters))
#         else:
#             self.b = None
#
#     def forward(self, x):
#         x = F.relu(self.gc(x, self.adj))
#         if self.bias:
#             x = x + self.b.unsqueeze(0)
#         cluster_labels = F.softmax(x, dim=1).argmax(dim=1)
#         return cluster_labels

import torch
from torch_geometric.nn import graclus

row = torch.arange(0, 10, 1).unsqueeze(1).repeat(1, 10).flatten()
col = torch.arange(0, 10, 1).unsqueeze(1).repeat(1, 10).permute(1, 0).flatten()
# print(row.size())
edge_index = torch.concat([row.unsqueeze(0), col.unsqueeze(0)], dim=0)
weight = torch.rand([100])
# print(edge_index.size())
labels = graclus(edge_index, weight, 10)
print(labels)
