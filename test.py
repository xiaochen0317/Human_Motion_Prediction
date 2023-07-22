import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
from utils.parser import args
import os
from model_copy import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)
model_name = 'h36_3d_' + str(args.output_n) + 'frames_ckpt'
model = Model(args.input_dim, args.hidden_features, args.input_n, args.output_n, args.st_gcnn_dropout,
              args.n_tcnn_layers, args.tcnn_kernel_size, args.tcnn_dropout, args.heads, args.alpha,
              args.spatial_scales, args.temporal_scales).to(device)

model.load_state_dict(torch.load(os.path.join(args.model_path, model_name)))
print(model)
