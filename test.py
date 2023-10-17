from __future__ import print_function, division
import sys

sys.path.append('core')
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import evaluate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import distributed as dist

# 改动1，获取进程号，用于分配GPU
local_rank = int(os.environ["LOCAL_RANK"])

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

SUM_FREQ = 100
VAL_FREQ = 5000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.05,
                                              cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)
        # print the training status
        # print(training_str + metrics_str)
        if self.writer is None:
            self.writer = SummaryWriter()
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        if self.total_steps % SUM_FREQ == SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):
    # 改动5，模型初始化方式，SyncBatchNorm为同步BN，可选
    device = torch.device('cuda:{}'.format(local_rank))
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(Mymodel(args)).to(device)
    model = Mymodel(args).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)

    print("Parameter Count: %d" % count_parameters(model))
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
    model.cuda()
    model.train()

    # 改动6，dataset加载，不用设置shuffle，其余可自己更改设置
    train_dataset = MyDataset(args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True,
                                   num_workers=4, drop_last=True, sampler=train_sampler)

    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)

    # 改动7，SummaryWriter只在单个进程记录数据，保存模型也是，下面不再一一注释
    if local_rank == 0:
        logger = Logger(model, scheduler)
    else:
        logger = None

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            data, label = [x.cuda() for x in data_blob]
            prediction = model(data)
            loss, metrics = compute_loss(prediction, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            if local_rank == 0:
                logger.push(metrics)
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps + 1, args.name)
                if local_rank == 0:
                    torch.save(model.state_dict(), PATH)
                    results = {}
                    for val_dataset in args.validation:
                        if val_dataset == 'validate_dataset':
                            results.update(evaluate.validate(model.module))
                    logger.write_dict(results)
                    model.train()
                    # 单进程测试，其他进程进入等待
                    torch.distributed.barrier()
            else:
                torch.distributed.barrier()
        total_steps += 1
        if total_steps > args.num_steps:
            should_keep_training = False
            break


if local_rank == 0:
    logger.close()
PATH = 'checkpoints/%s.pth' % args.name
torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    args = parser.parse_args()

    # 改动2，不同进程设置不同的随机种子
    torch.manual_seed(1234 + local_rank * 10)
    np.random.seed(1234 + local_rank * 10)
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    # 改动3，将batch_size变为1/N倍，N为卡的数量
    assert args.batch_size % torch.cuda.device_count() == 0
    args.batch_size = args.batch_size // torch.cuda.device_count()

    # 改动4，初始化进程组，分配GPU
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{}'.format(local_rank))

    train(args)

