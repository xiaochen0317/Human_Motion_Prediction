import torch

connect = [(2, 3), (3, 4), (4, 5), (7, 8), (8, 9), (9, 10), (17, 18), (18, 19), (21, 22), (25, 26), (26, 27), (29, 30),
           (14, 15)]


joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])

def get_spatial_connection(n_nodes, link):
    A = torch.zeros((n_nodes, n_nodes))
    for i, j in link:
        A[i, j] = 1