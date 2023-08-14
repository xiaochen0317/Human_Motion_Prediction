import torch

def calculate_edge_features(adjacency_matrix, node_features):
    # 计算边特征张量
    num_nodes = adjacency_matrix.size(0)
    batch_size = node_features.size(0)
    num_features = node_features.size(2)

    # 扩展邻接矩阵以适应节点特征集合的维度
    expanded_adj = adjacency_matrix.unsqueeze(2).expand(-1, -1, num_features)

    # 计算边特征张量
    edge_features = node_features.unsqueeze(1) - node_features.unsqueeze(2)
    edge_features *= expanded_adj

    return edge_features

# 示例邻接矩阵和节点特征集合（使用 PyTorch 张量）
adjacency_matrix = torch.tensor([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])

node_features = torch.tensor([[[1, 2],
                               [3, 4],
                               [5, 6]],
                              [[7, 8],
                               [9, 10],
                               [11, 12]]], dtype=torch.float32)

edge_features = calculate_edge_features(adjacency_matrix, node_features)

print("Edge Features Tensor:")
print(edge_features)
