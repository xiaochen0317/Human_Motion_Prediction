from torch.utils.data import Dataset
import numpy as np
from utils.DCT import get_dct_matrix, dct_transform_torch
from utils.parser import args
import utils.data_utils as data_utils


class MotionDataset(Dataset):
    def __init__(self, data_dir, actions, mode_name='train', input_n=10, output_n=25, split=0,
                 sample_rate=2, test_manner='256', global_max=0, global_min=0, device='cuda:0'):
        self.data_dir = data_dir
        self.split = split

        subs = [[1, 6, 7, 8, 9], [5], [11]]
        subs = [[1], [5], [11]]
        acts = data_utils.define_actions(actions)

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(data_dir, subjs, acts, sample_rate, input_n + output_n,
                                                                 device=device, test_manner=test_manner)

        gt_32 = all_seqs.transpose(0, 2, 1)  # B, T, 3*J -> B, 3*J, T
        gt_22 = gt_32[:, dim_used, :]

        if args.is_repeated:
            input_22 = np.concatenate((gt_22[:, :, :input_n], np.repeat(gt_22[:, :, input_n - 1:input_n], output_n,
                                                                        axis=-1)), axis=-1)
            self.dct_used = input_n + output_n
        else:
            input_22 = gt_22
            self.dct_used = input_n

        if args.is_dct:
            self.dct_m, self.idct_m = get_dct_matrix(input_n + output_n)
            input_22 = dct_transform_torch(input_22, self.dct_m, self.dct_used)

        self.global_max = global_max
        self.global_min = global_min

        if mode_name == 'train':
            self.global_max = np.max(input_22)
            self.global_min = np.min(input_22)

        input_22 = (input_22 - self.global_min) / (self.global_max - self.global_min)
        input_22 = input_22 * 2 - 1
        if args.is_norm:
            self.input_22 = input_22
        else:
            self.input_22 = gt_22

    def __len__(self):
        return self.input_22.shape[0]

    def __getitem__(self, item):
        return self.input_22


if __name__ == '__main__':
    pass
