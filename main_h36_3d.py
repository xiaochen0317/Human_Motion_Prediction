import os
from utils import h36motion3d as datasets
from torch.utils.data import DataLoader
from model import Model
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
from utils.parser import args
from bone_length import bone_length_loss


def seed_torch(seed=13000):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


seed_torch()


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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

model = Model(args.input_dim, args.hidden_features, args.input_n, args.output_n, args.st_attn_dropout,
              args.st_cnn_dropout, args.n_tcnn_layers, args.tcnn_kernel_size, args.tcnn_dropout, args.heads,
              args.spatial_scales, args.temporal_scales, args.temporal_scales2).to(device)


print('total number of parameters of the network is: ' + str(
    sum(p.numel() for p in model.parameters() if p.requires_grad)))
# print(model)
model_name = 'h36_3d_' + str(args.output_n) + 'frames_ckpt_test'


def train():
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)
    # optimizer_reverse = optim.Adam(model_reverse.parameters(), lr=args.lr, weight_decay=1e-05)
    # torch.autograd.set_detect_anomaly(True)

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    #     scheduler_loss = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_threshold = 80
    best_acc = 1e9

    train_loss = []
    val_loss = []
    dataset = datasets.Datasets(args.data_dir, args.input_n, args.output_n, args.skip_rate, split=0)
    print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    vald_dataset = datasets.Datasets(args.data_dir, args.input_n, args.output_n, args.skip_rate, split=1)
    print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
    vald_loader = DataLoader(vald_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                         46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                         75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    I_link = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 9, 12, 13, 14, 14, 9, 17, 18, 19, 19])
    J_link = np.array([1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

    for epoch in range(args.n_epochs):
        running_loss = 0
        # if epoch % 2 == 0:
        #     args.lr = lr_decay(optimizer, args.lr, args.lr_decay)
        n = 0
        model.train()
        print('lr: %.6f' % (optimizer.state_dict())['param_groups'][0]['lr'])
        for cnt, batch in enumerate(data_loader):
            batch = batch.to(device)
            batch_dim = batch.shape[0]
            n += batch_dim

            # 改变
            sequences_train = batch[:, 0:args.input_n, dim_used].view(-1, args.input_n, len(dim_used) // 3, 3).\
                permute(0, 3, 1, 2)
            sequences_gt = batch[:, args.input_n:args.input_n + args.output_n, dim_used].view(-1, args.output_n,
                                                                                              len(dim_used) // 3, 3)

            optimizer.zero_grad()
            # optimizer_reverse.zero_grad()

            sequences_predict = model(sequences_train)
            sequences_predict = sequences_predict.permute(0, 1, 3, 2)  # B, T， J， 3

            loss1 = mpjpe_error(sequences_predict, sequences_gt)
            # loss2 = bone_length_loss(sequences_train.permute(0, 2, 3, 1),
            #                          sequences_predict, I_link, J_link)

            # loss = loss1 + args.loss_parameter * loss2 + l + 0.1*e
            loss = loss1
            # print('loss: %.3f, error: %.3f' % (loss1, e))

            if cnt % 200 == 0:
                print('[%d, %5d]  training loss: %.3f' % (epoch + 1, cnt + 1, loss1.item()))
                # print('loss: %.3f, l: %.3f, e: %.3f' % (loss1, l, e))

            # loss_reverse.backward(retain_graph=True)
            # optimizer_reverse.step()

            loss.backward()
            # optimizer_reverse.step()
            optimizer.step()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            running_loss += loss1 * batch_dim

        train_loss.append(running_loss.detach().cpu() / n)

        model.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0
            for cnt, batch in enumerate(vald_loader):
                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                sequences_train = batch[:, 0:args.input_n, dim_used].view(-1, args.input_n, len(dim_used) // 3, 3).\
                permute(0, 3, 1, 2)
                sequences_gt = batch[:, args.input_n:args.input_n + args.output_n, dim_used].view(-1, args.output_n,
                                                                                                  len(dim_used) // 3, 3)

                sequences_predict = model(sequences_train)
                sequences_predict = sequences_predict.permute(0, 1, 3, 2)  # B, T, J, 3
                loss1 = mpjpe_error(sequences_predict, sequences_gt)
                # loss2 = bone_length_loss(sequences_train.permute(0, 2, 3, 1),
                #                          sequences_predict, I_link, J_link)
                # loss = loss1 + args.loss_parameter * loss2 + e

                if cnt % 200 == 0:
                    print('[%d, %5d]  validation loss: %.3f' % (epoch + 1, cnt + 1, loss1.item()))
                running_loss += loss1 * batch_dim
            val_loss.append(running_loss.detach().cpu() / n)
            # print(n)
        test_error = test()
        print('epoch %d  training loss: %.3f, validation loss: %.3f, test: %.3f' %
              (epoch + 1, train_loss[-1], val_loss[-1], test_error))

        # if (epoch + 1) % 50 == 0:
        #     plt.figure(1)
        #     plt.plot(train_loss, 'r', label='Train loss')
        #     plt.plot(val_loss, 'g', label='Val loss')
        #     plt.legend()
        #     plt.show()
        if args.use_scheduler:
            scheduler.step()

        if test_error < best_acc:
            print('----saving model-----')
            torch.save(model.state_dict(), os.path.join(args.model_path, model_name))
            best_acc = test_error
        print('best loss: %.3f' % best_acc)


def test():
    # print(args.model_path)
    # print(model_name)
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
        dataset_test = datasets.Datasets(args.data_dir, args.input_n, args.output_n, args.skip_rate, split=2,
                                         actions=[action])
        print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

        test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, shuffle=False, num_workers=0,
                                 pin_memory=True)
        for cnt, batch in enumerate(test_loader):
            with torch.no_grad():
                batch = batch.to(device)
                batch_dim = batch.shape[0]
                n += batch_dim

                all_joints_seq = batch.clone()[:, args.input_n:args.input_n + args.output_n, :]

                sequences_train = batch[:, 0:args.input_n, dim_used].view(-1, args.input_n, len(dim_used) // 3, 3).\
                permute(0, 3, 1, 2)
                sequences_gt = batch[:, args.input_n:args.input_n + args.output_n, :]

                sequences_predict = model(sequences_train)
                sequences_predict = sequences_predict.permute(0, 1, 3, 2).contiguous().view(-1, args.output_n,
                                                                                            len(dim_used))

                all_joints_seq[:, :, dim_used] = sequences_predict

                all_joints_seq[:, :, index_to_ignore] = all_joints_seq[:, :, index_to_equal]
                # loss = mpjpe_error(all_joints_seq.view(-1, args.output_n, 32, 3),
                #                    sequences_gt.view(-1, args.output_n, 32, 3))
                loss = mpjpe_error(all_joints_seq[:, -1].view(-1, 1, 32, 3),
                                   sequences_gt[:, -1].view(-1, 1, 32, 3))
                running_loss += loss * batch_dim
                accum_loss += loss * batch_dim

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
        visualize(args.input_n, args.output_n, args.visualize_from, args.data_dir, model, device, args.n_viz,
                  args.skip_rate, args.actions_to_consider)
