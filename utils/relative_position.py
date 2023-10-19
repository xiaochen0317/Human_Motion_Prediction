import numpy as np
import networkx as nx
import torch.nn as nn
import torch


def cal_spatial_adj(I_link=None, J_link=None, joints=22, self_connected=False):
    if I_link is None:
        I_link = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
    if J_link is None:
        J_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    A = np.zeros([joints, joints])
    for i in range(len(I_link)):
        A[I_link[i], J_link[i]] = 1
        A[J_link[i], I_link[i]] = 1
    if self_connected is True:
        for j in range(joints):
            A[j, j] = 1
    return A


def cal_temporal_adj(frames=10, self_connected=False):
    A = np.zeros([frames, frames])
    for i in range(frames - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
        if self_connected is True:
            A[i, i] = 1
    return A


def cal_SPD(nodes, adj):
    G = nx.Graph()
    G.add_nodes_from(range(nodes))
    for i in range(nodes):
        for j in range(nodes):
            if adj[i][j] == 1:
                G.add_edge(i, j)
    shortest_path = []
    for i in range(nodes):
        shortest_path.append([])
        for j in range(nodes):
            shortest_path[i].append(nx.shortest_path(G, source=i, target=j))
    return shortest_path


def cal_ST_SPD(joints=22, frames=10, I22_link=None, J22_link=None):
    if I22_link is None:
        I22_link = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
    if J22_link is None:
        J22_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    S_adj = cal_spatial_adj(I22_link, J22_link, joints)
    T_adj = cal_temporal_adj(frames)
    S_SPD = cal_SPD(joints, S_adj)
    T_SPD = cal_SPD(frames, T_adj)
    return S_SPD, T_SPD


class Spatial_Edge_Enhance(nn.Module):
    def __init__(self, joints, frames, I22_link, J22_link, embedding_dim):
        super(Spatial_Edge_Enhance, self).__init__()
        self.joints = joints
        self.frames = frames
        self.I22_link = I22_link
        self.J22_link = J22_link
        self.embedding_dim = embedding_dim

        # 定义您需要的层或操作
        self.linear = nn.Linear(embedding_dim, embedding_dim)  # 示例线性层

    def forward(self, src):
        # src: B x N x C

        # 计算空间和时间最短路径矩阵
        S_SPD, T_SPD = cal_ST_SPD(self.joints, self.frames, self.I22_link, self.J22_link)

        # 使用最短路径矩阵计算两两节点的关系
        pairwise_relations = self.calculate_pairwise_relations(S_SPD, T_SPD, src)

        # 对关系应用任何必要的层或操作
        pairwise_relations = self.linear(pairwise_relations)  # 示例线性层

        return pairwise_relations

    def calculate_pairwise_relations(self, S_SPD, T_SPD, src):
        # 基于最短路径矩阵计算两两节点的关系
        pairwise_relations = torch.zeros(self.joints, self.joints, self.embedding_dim)

        for i in range(self.joints):
            for j in range(self.joints):
                cumulative_embedding = torch.zeros(self.embedding_dim)
                for path in S_SPD[i][j]:
                    for k in range(len(path) - 1):
                        edge_embedding = src[:, path[k + 1], :] - src[:, path[k], :]
                        cumulative_embedding += edge_embedding
                pairwise_relations[i, j, :] = cumulative_embedding

        return pairwise_relations