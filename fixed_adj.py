import numpy as np
import torch
from torch.utils.data import DataLoader
import os

import utils.h36motion3d as datasets
from utils.data_utils import define_actions
from utils.parser import args
# from model_copy import Model

# H36M
I22_link = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
J22_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])


def spatial_fixed_adj(joints, frames):
    # H36M
    I_link = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
    J_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    # I = torch.from_numpy(I)
    # J = torch.from_numpy(J)
    s_adj = torch.zeros([joints, joints])
    for i in range(len(I_link)):
        s_adj[I_link[i], J_link[i]] = 1
        s_adj[J_link[i], I_link[i]] = 1
    for j in range(joints):
        s_adj[j, j] = 1
    # s_adj.unsqueeze(0).repeat(frames, 1, 1)
    return s_adj


def temporal_fixed_adj(joints, frames):
    t_adj = torch.zeros([frames, frames])
    for i in range(frames - 1):
        t_adj[i, i + 1] = 1
        t_adj[i + 1, i] = 1
    # t_adj.unsqueeze(0).repeat(joints, 1, 1)
    return t_adj


if __name__ == '__main__':
    print(1)
