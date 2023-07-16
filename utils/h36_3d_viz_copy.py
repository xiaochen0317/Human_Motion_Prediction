#!/usr/bin/env python
# coding: utf-8
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import utils.h36motion3d as datasets
from utils.loss_funcs import mpjpe_error
from utils.data_utils import define_actions


def draw_pose(ax, vals, pred=True):
    # h36m 32 joints(full)
    connect = [
        (1, 2), (2, 3), (3, 4), (4, 5),
        (6, 7), (7, 8), (8, 9), (9, 10),
        (0, 1), (0, 6),
        (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
        (1, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        (24, 25), (24, 17),
        (24, 14), (14, 15)
    ]
    LR = [
        False, True, True, True, True,
        True, False, False, False, False,
        False, True, True, True, True,
        True, True, False, False, False,
        False, False, False, False, True,
        False, True, True, True, True,
        True, True
    ]

    # Start and endpoints of our representation
    I = np.array([touple[0] for touple in connect])
    J = np.array([touple[1] for touple in connect])  # 选取了其中18个关节点一半左一半右
    # Left / right indicator
    LR = np.array([LR[a] or LR[b] for a, b in connect])
    if pred:
        lcolor = "#9b59b6"
        rcolor = "#2ecc71"
    else:
        lcolor = "#8e8e8e"
        rcolor = "#383838"

    for i in np.arange(len(I)):
        x = np.array([vals[I[i], 0], vals[J[i], 0]])
        z = np.array([vals[I[i], 1], vals[J[i], 1]])
        y = np.array([vals[I[i], 2], vals[J[i], 2]])
        print(i,x,y,z)
        if i == 0:
            ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor)
                    # label=['GT' if not pred else 'Pred'])
        else:
            ax.plot(x, y, z, lw=2, linestyle='--', c=lcolor if LR[i] else rcolor)
    plt.pause(0.1)


def visualize(input_n, output_n, visualize_form, path, modello, device, n_viz, skip_rate, actions):
    actions = define_actions("walking")

    for action in actions:

        if visualize_form == 'train':
            loader = datasets.Datasets(path, input_n, output_n, skip_rate, split=0, actions=[action])
        elif visualize_form == 'validation':
            loader = datasets.Datasets(path, input_n, output_n, skip_rate, split=1, actions=[action])
        elif visualize_form == 'test':
            loader = datasets.Datasets(path, input_n, output_n, skip_rate, split=2, actions=[action])

        dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                             26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                             46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                             75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
        # joints at same loc
        joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([13, 19, 22, 13, 27, 30])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        loader = DataLoader(
            loader,
            batch_size=1,
            shuffle=False,  # for comparable visualizations with other models
            num_workers=0)

        for cnt, batch in enumerate(loader):
            batch = batch.to(device)

            all_joints_seq = batch.clone()[:, input_n:input_n + output_n, :]

            sequences_train = batch[:, 0:input_n, dim_used].view(-1, input_n, len(dim_used) // 3, 3).permute(0, 3, 1, 2)
            sequences_gt = batch[:, input_n:input_n + output_n, :]

            sequences_predict = modello(sequences_train).permute(0, 1, 3, 2).contiguous().view(-1, output_n,
                                                                                               len(dim_used))

            all_joints_seq[:, :, dim_used] = sequences_predict

            all_joints_seq[:, :, index_to_ignore] = all_joints_seq[:, :, index_to_equal]

            all_joints_seq = all_joints_seq.view(-1, output_n, 32, 3)

            sequences_train = batch[:, 0:input_n, :].view(-1, input_n, 32, 3)
            sequences_gt = sequences_gt.view(-1, output_n, 32, 3)

            loss = mpjpe_error(all_joints_seq, sequences_gt)  # # both must have format (batch,T,V,C)

            data_train = torch.squeeze(sequences_train, 0).cpu().data.numpy() / 1000
            data_pred = torch.squeeze(all_joints_seq, 0).cpu().data.numpy() / 1000  # in meters
            data_gt = torch.squeeze(sequences_gt, 0).cpu().data.numpy() / 1000

            fig = plt.figure()
            # ax = Axes3D(fig)
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=20, azim=-40)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            # ax.legend(loc='lower left')

            ax.set_xlim3d([-1, 1.5])
            ax.set_xlabel('X')

            ax.set_ylim3d([-1, 1.5])
            ax.set_ylabel('Y')

            ax.set_zlim3d([0.0, 1.5])
            ax.set_zlabel('Z')
            ax.set_title(
                'loss in mm is: ' + str(round(loss.item(), 4)) + ' for action : ' + str(action) + ' for ' + str(
                    output_n) + ' frames')
            plt.pause(0.1)
            for i in range(input_n):
                draw_pose(ax, data_train[i, :, :], pred=False)
                print(i)

            for i in range(output_n):
                draw_pose(ax, data_gt[i, :, :], pred=False)
                draw_pose(ax, data_pred[i, :, :], pred=True)
            plt.show()
            if cnt == n_viz - 1:
                break
