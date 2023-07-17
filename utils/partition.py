import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import SpectralClustering


def spectral_clustering(A, num_clusters):
    """
    :param A: 邻接矩阵
    :param num_clusters: 类的数量
    :return: 节点的社区标签，类型为numpy数组
    """
    # A = A.detach().numpy()
    A = (A + A.transpose(1, 2)) / 2  # 使邻接矩阵对称
    D = torch.sum(A, dim=1)
    D_sqrt = torch.sqrt(torch.clamp(D, min=1e-9))
    D_inv_sqrt = 1.0 / D_sqrt
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    L = torch.diag(D_inv_sqrt) @ A @ torch.diag(D_inv_sqrt)

    eigenvalues, eigenvectors = torch.eig(L, eigenvectors=True)

    _, indices = torch.sort(eigenvalues[:, 0])
    eigenvectors = eigenvectors[: ,indices]

    k_eigenvectors = eigenvectors[:, :num_clusters]
    k_eigenvectors_norm = nn.functional.normalize(k_eigenvectors, dim=-1)
    kmeans = torch.cluster.kMeans(num_clusters)
    labels = kmeans.fit(k_eigenvectors_norm)

    return labels

def partition_and_pooling(src, adj, num_clusters, mode):
    """
    :param src: 输入数据 [B*J, T, C] / [B*C, J, C]
    :param adj: 时空邻接矩阵 [B*T, J, J] / [B*J, T, T]
    :param num_clusters: 类数量
    :param mode: 模式（时/空）
    :return: 经过处理的输出数据
    """
    if mode == 'spatial':  # adj: B*T, J, J
        B, J, C = src.size()
        labels = np.zeros((B, J))
        output = torch.zeros((B, num_clusters, C))
        for i in range(B):
            labels[i, :] = spectral_clustering(adj[i, :, :], num_clusters)
            # average information of joints in the same cluster
            unique_labels = np.unique(labels[i, :])
            # 遍历每个社区标签
            for label in unique_labels:
                indices = np.where(labels == label)[0]
                output[i, label, :] = torch.mean(src[i, indices, :], dim=-1)
    else:
        B, T, C = src.size()
        labels = np.zeros((B, T))
        output = torch.zeros((B, num_clusters, C))
        for i in range(B):
            labels[i, :] = spectral_clustering(adj[i, :, :], num_clusters)
            # average information of joints in the same cluster
            unique_labels = np.unique(labels[i, :])
            # 遍历每个社区标签
            for label in unique_labels:
                # 找到所有标签为label的节点
                indices = np.where(labels == label)[0]
                # 计算这些节点的平均信息
                output[i, label, :] = torch.mean(src[i, indices, :], dim=-1)
    return output  # B*T, C, N / B*J, C, N


if __name__ == '__main__':
    # 随机生成一个5x5的邻接矩阵
    Ab = torch.rand((5, 5))
    print('邻接矩阵：\n', Ab)

    # 进行图分割，得到每个节点的社区标签
    num_clustersb = 2
    labelb = spectral_clustering(Ab, num_clustersb)
    print('社区标签：', labelb)
