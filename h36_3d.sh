#!/bin/bash
mkdir -p checkpoints
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port 8888 main_h36_3d_parallel.py --batch_size 256
