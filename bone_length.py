import numpy as np
import torch
from torch.utils.data import DataLoader
import os

import utils.h36motion3d as datasets
from utils.data_utils import define_actions
from utils.parser import args
from model_copy import Model


def cal_bone_length_test(input_n, output_n, visualize_from, path, model, device, n_viz, skip_rate):
    actions = define_actions("walking")
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    I22_link = np.array([8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
    J22_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

    for action in actions:

        if visualize_from == 'train':
            loader = datasets.Datasets(path, input_n, output_n, skip_rate, split=0, actions=[action])
        elif visualize_from == 'validation':
            loader = datasets.Datasets(path, input_n, output_n, skip_rate, split=1, actions=[action])
        elif visualize_from == 'test':
            loader = datasets.Datasets(path, input_n, output_n, skip_rate, split=2, actions=[action])

        loader = DataLoader(
            loader,
            batch_size=1,
            shuffle=False,  # for comparable visualizations with other models
            num_workers=0)

        for cnt, batch in enumerate(loader):
            # 22 joints
            batch = batch[:, 0:input_n+output_n, dim_used]
            bone_len = np.zeros(len(I22_link))
            batch = batch.view(-1, input_n+output_n, len(dim_used) // 3, 3).squeeze(0)  # T+K, J, 3
            for i in range(len(I22_link)):
                front = batch[:, I22_link[i], :]
                end = batch[:, J22_link[i], :]
                bone_len[i] = np.mean(np.linalg.norm(front - end, axis=1))
                print('Bone %d length:%.3f mm' % (i, bone_len[i]))
                print(np.linalg.norm(front - end, axis=1))
            if cnt == n_viz - 1:
                break


def cal_bone_length(input_data, I_link, J_link):
    """
    Args:
        input_data: [batch_size, input_frame, joints_number, dimension], input data
        I_link: [bone_number], head joint index
        J_link: [bone_number], end joint index

    Returns:
        output_bone_length: [batch_size, bone_number], length collection of each bone

    """

    # I_link = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
    # J_link = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    B, T, V, C = input_data.size()
    output_bone_length = torch.zeros((B, len(I_link)))
    for i in range(len(I_link)):
        head_joint = input_data[:, :, I_link[i], :]
        end_joint = input_data[:, :, J_link[i], :]
        output_bone_length[:, i] = torch.mean((torch.norm(head_joint - end_joint, dim=-1)), dim=1)
    return output_bone_length


def bone_length_loss(input_data, pred_data, I_link, J_link):
    """
    Args:
        input_data: [batch_size, input_frame, joints_number, dimension], input training data
            pred_data: [batch_size, output_frame, joints_number, dimension], predicted output data
        I_link: [bone_number], head joint index
        J_link: [bone_number], end joint index

    Returns:
        loss: length of bone-length variation

    """
    # I_link = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
    # J_link = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    B, K, V, C = pred_data.size()
    bone_length = cal_bone_length(input_data, I_link, J_link)
    pred_bone_length = cal_bone_length(pred_data, I_link, J_link)
    loss = torch.mean(torch.norm(pred_bone_length - bone_length, dim=-1))
    return loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)
    model = Model(args.input_dim, args.input_n, args.output_n, args.st_gcnn_dropout,
                  args.n_tcnn_layers, args.tcnn_kernel_size, args.tcnn_dropout, args.d_hid, args.joints_to_consider) \
        .to(device)
    model_name = 'h36_3d_' + str(args.output_n) + 'frames_ckpt'
    model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
    model.eval()
    cal_bone_length_test(args.input_n, args.output_n, args.visualize_from, args.data_dir, model, device, 1,
                         args.skip_rate)
