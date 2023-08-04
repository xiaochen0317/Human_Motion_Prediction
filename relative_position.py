import numpy as np
import networkx as nx
import torch.nn as nn


def cal_spatial_adj(I_link, J_link, joints, self_connected=False):
    A = np.zeros([joints, joints])
    for i in range(len(I_link)):
        A[I_link[i], J_link[i]] = 1
        A[J_link[i], I_link[i]] = 1
    if self_connected is True:
        for j in range(joints):
            A[j, j] = 1
    return A


def cal_temporal_adj(frames, self_connected=False):
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

