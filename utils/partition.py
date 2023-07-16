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
    A = A.cpu().numpy()
    A = (A + A.T) / 2  # 使邻接矩阵对称

    # 创建谱聚类模型
    model = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
    labels = model.fit_predict(A)

    return labels


def partition_and_pooling(src, adj, num_clusters, mode):
    """
    :param src: 输入数据 [B, C, T ,J]
    :param adj: 时空邻接矩阵 [B, M, N, N]
    :param num_clusters: 类数量
    :param mode: 模式（时/空）
    :return: 经过处理的输出数据
    """
    B, C, T, J = src.size()
    if mode == 'spatial':  # adj: B, T, J, J
        adj = adj.permute(B * T, J, J)
        labels = np.zeros((B * T, J))
        output = torch.zeros((B, C, T, num_clusters))
        for i in range(B * T):
            labels[i, :] = spectral_clustering(adj[i, :, :], num_clusters)
        labels = labels.reshape((B, T, J))
        # average information of joints in the same cluster
        unique_labels = np.unique(labels)
        # 遍历每个社区标签
        for label in unique_labels:
            # 找到所有标签为label的节点
            indices = np.where(labels == label)[0]
            # 计算这些节点的平均信息
            output[:, :, :, label] = torch.mean(src[indices], dim=3)
    else:
        adj = adj.permute(B * J, T, T)
        labels = np.zeros((B * J, T))
        output = torch.zeros((B, C, num_clusters, J))
        for i in range(B * J):
            labels[i, :] = spectral_clustering(adj[i, :, :], num_clusters)
        labels = labels.reshape((B, J, T))
        # average information of joints in the same cluster
        unique_labels = np.unique(labels)
        # 遍历每个社区标签
        for label in unique_labels:
            # 找到所有标签为label的节点
            indices = np.where(labels == label)[0]
            # 计算这些节点的平均信息
            output[:, :, label, :] = torch.mean(src[indices], dim=2)
    return output  # B, C, T, N / B, C, N, J


if __name__ == '__main__':
    # 随机生成一个5x5的邻接矩阵
    Ab = torch.rand((5, 5))
    print('邻接矩阵：\n', Ab)

    # 进行图分割，得到每个节点的社区标签
    num_clustersb = 2
    labelb = spectral_clustering(Ab, num_clustersb)
    print('社区标签：', labelb)
