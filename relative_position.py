import torch.nn as nn


def RPE(A):
    h1 = A.sum(dim=0)  # 骨架图的度向量
    h1[h1 != 0] = 1
