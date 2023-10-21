import argparse
import os

parser = argparse.ArgumentParser(description='Arguments for running the scripts')
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--restore_ckpt", type=str, default=None)
parser.add_argument("--mixed_precision", type=bool, default=True)

# ARGS FOR LOADING THE DATASET
parser.add_argument('--data_dir', type=str, default='/home/yons/fuxian/h3.6m',
                    help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5],
                    help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
parser.add_argument('--joints_to_consider', type=int, default=22, choices=[16, 18, 22],
                    help='number of joints to use, defaults=16 for H36M angles, 22 for H36M 3D or 18 for AMASS/3DPW')
parser.add_argument('--is_normalized', type=bool, default=False)
parser.add_argument('--is_dct', type=bool, default=False)
parser.add_argument('--is_repeated', type=bool, default=False)
# todo: 因为我们需要考虑节点之间的时空关联特性，如果进行DCT，那么节点之间的时空关系可能就发生变化了？（但是实际上，在除了第一层的模块，
#  其他的模型都不是最初始的特征。）

# ARGS FOR THE MODEL
parser.add_argument('--n_encoder_layer', type=int, default=4, help='number of encoder layers')
parser.add_argument('--n_decoder_layer', type=int, default=6, help='number of decoder layers')
parser.add_argument('--dim', type=int,  default=64, help='dimensions for the coordinates of the embedding')
parser.add_argument('--dcn_n', type=int, default=10, help='coefficient of dct')
parser.add_argument('--input_dim', type=int, default=9, help='dimensions of the input coordinates')
parser.add_argument('--output_dim', type=int, default=3, help='dimensions of the output coordinates')
parser.add_argument('--dropout', type=float, default=0.1, help='mlp dropout')
parser.add_argument('--attn_dropout', type=float, default=0.1, help='attn dropout')
parser.add_argument('--tcn_dropout', type=float, default=0.0, help='tcn dropout')
parser.add_argument('--drop_path', type=float, default=0.0, help='drop_path')
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--qk_bias', type=bool, default=False)
parser.add_argument('--qk_scale', type=int, default=None)
parser.add_argument('--spatial_scales', type=list, default=[22])
parser.add_argument('--temporal_scales', type=list, default=[10])
# parser.add_argument('--spatial_scales', type=list, default=[22, 12, 5])
# parser.add_argument('--temporal_scales', type=list, default=[10, 6, 3])
parser.add_argument('--lr_decay', type=float, default=1)
parser.add_argument('--alpha', type=float, default=0.5)

# ARGS FOR THE TRAINING

parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'viz'],
                    help='Choose to train,test or visualize from the model.Either train,test or viz')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
parser.add_argument('--lr', type=int, default=1e-2, help='Learning rate of the optimizer')
parser.add_argument('--use_scheduler', type=bool, default=True, help='use MultiStepLR scheduler')
parser.add_argument('--milestones', type=list, default=[30, 40, 50],
                    help='the epochs after which the learning rate is adjusted by gamma')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='gamma correction to the learning rate, after reaching the milestone epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
parser.add_argument('--model_path', type=str, default='./checkpoints/CKPT_3D_H36M',
                    help='directory with the models checkpoints ')

# FLAGS FOR THE VISUALIZATION

parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'],
                    help='choose data split to visualize from(train-val-test)')
parser.add_argument('--actions_to_consider', default='all',
                    help='Actions to visualize.Choose either all or a list of actions')
parser.add_argument('--n_viz', type=int, default='2', help='Numbers of sequences to visualize for each action')


args = parser.parse_args()
