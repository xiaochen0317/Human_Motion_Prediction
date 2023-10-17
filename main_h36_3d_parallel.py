import os
import random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.autograd
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed
from torch.cuda.amp import GradScaler
from model_modify import Model
import matplotlib.pyplot as plt
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
from utils.parser import args
from utils import h36motion3d as datasets
from tqdm import tqdm


def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def fine_tuning(optimizer, lr_ft, loss):
    if loss <= 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_ft
    return lr_ft


def lr_decay(optimizer, lr_now, gamma):
    lr = lr_now * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


local_rank = int(os.environ["LOCAL_RANK"])
print(local_rank)
seed_torch()

if local_rank != -1:
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

device = torch.device('cuda:{}'.format(local_rank))
model = Model(args.input_dim, args.output_dim, args.dim, args.heads, args.spatial_scales, args.temporal_scales,
              args.qk_bias, args.qk_scale, args.dropout, args.attn_dropout, args.tcn_dropout, args.drop_path,
              args.input_n, args.output_n, args.joints_to_consider, args.dcn_n, args.n_encoder_layer,
              args.n_decoder_layer).to(device)

model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                            find_unused_parameters=True)

print('Total number of parameters is: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))
if args.restore_ckpt is not None:
    model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
print(model)
model_name = 'h36_3d_' + str(args.output_n) + 'frames_ckpt_test'

spatial_edge_index = torch.tensor([
    [8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19,
     0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
     8, 0, 1, 2, 8, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19]])
temporal_edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 7, 8,
     1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 5, 6, 7, 8, 9,
     0, 1, 2, 3, 4, 5, 6, 7, 8]])
spatial_adj = torch.sparse_coo_tensor(spatial_edge_index, torch.ones(spatial_edge_index.shape[1]),
                                      torch.Size([args.joints_to_consider, args.joints_to_consider])) \
    .to_dense().cuda(non_blocking=True)
temporal_adj = torch.sparse_coo_tensor(temporal_edge_index, torch.ones(temporal_edge_index.shape[1]),
                                       torch.Size([args.input_n, args.input_n])).to_dense().cuda(non_blocking=True)


def train():
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    else:
        scheduler = None
    best_acc = 1e9

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)

    train_loss = []
    val_loss = []
    train_dataset = datasets.Datasets(args.data_dir, args.input_n, args.output_n, args.skip_rate, split=0)
    print('>>> Training dataset length: {:d}'.format(train_dataset.__len__()))
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valid_dataset = datasets.Datasets(args.data_dir, args.input_n, args.output_n, args.skip_rate, split=1)
    print('>>> Validation dataset length: {:d}'.format(valid_dataset.__len__()))
    valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

    for epoch in range(args.n_epochs):
        running_loss = 0
        train_sampler.set_epoch(epoch)
        n = 0
        model.train()
        print('lr: %.8f' % (optimizer.state_dict())['param_groups'][0]['lr'])
        for cnt, batch in enumerate(tqdm(train_loader)):
            batch = batch.cuda(non_blocking=True)
            batch_dim = batch.shape[0]
            n += batch_dim

            sequences_train = batch[:, 0:args.input_n, dim_used].view(-1, args.input_n, len(dim_used) // 3, 3)
            sequences_gt = batch[:, args.input_n:args.input_n + args.output_n, dim_used].view(-1, args.output_n,
                                                                                              len(dim_used) // 3, 3)
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()

                # sequences_predict, sm_loss_all, so_loss_all, tm_loss_all, to_loss_all = model(sequences_train)
                sequences_predict = model(sequences_train, spatial_adj, temporal_adj)
                sequences_predict = sequences_predict

                loss1 = mpjpe_error(sequences_predict, sequences_gt)
                loss = loss1

                if cnt % 200 == 199:
                    print('[%d, %5d]  training loss: %.3f' % (epoch + 1, cnt + 1, loss1.item()))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # for name, param in model.named_parameters():
                #     if param.grad is None:
                #         print(name)
                if args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                if args.use_scheduler:
                    scheduler.step()
                scaler.update()
            running_loss += loss1 * batch_dim
        train_loss.append(running_loss.detach().cpu() / n)

        model.eval()
        if local_rank == 0:
            with torch.no_grad():
                running_loss = 0
                n = 0
                for cnt, batch in enumerate(tqdm(valid_loader)):
                    batch = batch.cuda(non_blocking=True)
                    batch_dim = batch.shape[0]
                    n += batch_dim

                    sequences_train = batch[:, 0:args.input_n, dim_used].view(-1, args.input_n, len(dim_used) // 3, 3)
                    sequences_gt = batch[:, args.input_n:args.input_n + args.output_n, dim_used]\
                        .view(-1, args.output_n, len(dim_used) // 3, 3)

                    # sequences_predict, sm_loss_all, so_loss_all, tm_loss_all, to_loss_all = model(sequences_train)
                    sequences_predict = model(sequences_train, spatial_adj, temporal_adj)
                    loss1 = mpjpe_error(sequences_predict, sequences_gt)
                    if cnt % 200 == 199:
                        print('[%d, %5d]  validation loss: %.3f' % (epoch + 1, cnt + 1, loss1.item()))
                    running_loss += loss1 * batch_dim
                val_loss.append(running_loss.detach().cpu() / n)
            torch.distributed.barrier()

        test_error = test()
        print('epoch %d  training loss: %.3f, validation loss: %.3f, testing loss: %.3f' %
              (epoch + 1, train_loss[-1], val_loss[-1], test_error))

        if (epoch + 1) % 20 == 0:
            plt.figure(1)
            plt.plot(train_loss, 'r', label='Train loss')
            plt.plot(val_loss, 'g', label='Val loss')
            plt.legend()
            plt.show()
        if test_error < best_acc:
            print('----saving model-----')
            if local_rank == 0:
                torch.save(model.state_dict(), os.path.join(args.model_path, model_name))
                best_acc = test_error
        print('best loss: %.3f' % best_acc)


def test():
    # model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
    model.eval()
    # print(model)
    accum_loss = 0
    n_batches = 0  # number of batches for all the sequences
    actions = define_actions(args.actions_to_consider)
    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

    for action in actions:
        running_loss = 0
        n = 0
        test_dataset = datasets.Datasets(args.data_dir, args.input_n, args.output_n, args.skip_rate, split=2,
                                         actions=[action])
        print('>>> test action for sequences: {:d}'.format(test_dataset.__len__()))
        test_sampler = DistributedSampler(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test, sampler=test_sampler,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():
                batch = batch.cuda(non_blocking=True)
                batch_dim = batch.shape[0]
                n += batch_dim

                all_joints_seq = batch.clone()[:, args.input_n:args.input_n + args.output_n, :]

                sequences_train = batch[:, 0:args.input_n, dim_used].view(-1, args.input_n, len(dim_used) // 3, 3)
                sequences_gt = batch[:, args.input_n:args.input_n + args.output_n, :]

                # sequences_predict, sm_loss_all, so_loss_all, tm_loss_all, to_loss_all = model(sequences_train)
                sequences_predict = model(sequences_train, spatial_adj, temporal_adj)
                sequences_predict = sequences_predict.view(-1, args.output_n, len(dim_used))

                all_joints_seq[:, :, dim_used] = sequences_predict

                all_joints_seq[:, :, index_to_ignore] = all_joints_seq[:, :, index_to_equal]
                # loss = mpjpe_error(all_joints_seq.view(-1, args.output_n, 32, 3),
                #                    sequences_gt.view(-1, args.output_n, 32, 3))
                # loss = mpjpe_error(all_joints_seq[:, :, dim_used].view(-1, args.output_n, 22, 3),
                #                    sequences_gt[:, :, dim_used].view(-1, args.output_n, 22, 3))
                loss = mpjpe_error(all_joints_seq[:, -1].view(-1, 1, 32, 3),
                                   sequences_gt[:, -1].view(-1, 1, 32, 3))
                running_loss += loss * batch_dim
                accum_loss += loss * batch_dim
                print(n)

        print('loss at test subject for action : ' + str(action) + ' is: ' + str(running_loss / n))
        n_batches += n
    # print(n_batches)
    print('overall average loss in mm is: ' + str(accum_loss / n_batches))
    return accum_loss / n_batches


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'viz':
        model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
        model.eval()
        visualize(args.input_n, args.output_n, args.visualize_from, args.data_dir, model, 'cuda:0', args.n_viz,
                  args.skip_rate, args.actions_to_consider)
