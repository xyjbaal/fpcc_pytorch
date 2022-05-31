#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40
from model import DGCNN_cls, convert_groupandcate_to_one_hot
import numpy as np
from torch.utils.data import DataLoader
from utils import IOStream
import sklearn.metrics as metrics

import provider
from config import args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb


D_MAX = args.d_max
vdm = args.use_vdm
asm = args.use_asm
MARGINS = [args.margin_same, args.margin_diff]

NUM_GROUPS = args.group_num

PRETRAINED_MODEL_PATH = os.path.join(args.model_path, 'ring_vdm_asm/')
MODEL_STORAGE_PATH = PRETRAINED_MODEL_PATH
if not os.path.exists(MODEL_STORAGE_PATH):
    os.mkdir(MODEL_STORAGE_PATH)

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')

def train(args, io):

    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    #Try to load models
    if args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    print(str(model))
    criterion = model.get_loss

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        total_loss = 0.0
        total_score_loss = 0.0
        total_grouperr = 0.0
        batch_count = 0
        model.train()
        for data, label, group in train_loader:
            group, _ = convert_groupandcate_to_one_hot(group.numpy(), NUM_GROUPS=NUM_GROUPS)
            group = torch.from_numpy(group)
            data, label, group = data.to(device), label.to(device).squeeze(), group.to(device).squeeze()
            labels = {'ptsgroup': group, 'center_score': label}
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            net_output = model(data)
            loss_val, score_loss_val, grouperr_val = criterion(net_output, labels, vdm, asm, D_MAX, MARGINS)
            loss_val.backward()
            opt.step()

            total_loss += loss_val
            total_score_loss += score_loss_val
            total_grouperr += grouperr_val
            batch_print = 'Batch: %d, loss: %f, score_loss: %f' % (batch_count, loss_val, score_loss_val)
            io.cprint(batch_print)
            if batch_count % 100 == 99:
                batch_print = 'Batch: %d, loss: %f, score_loss: %f, grouperr: %f' % (
                batch_count, total_loss / 100, total_score_loss / 100, total_grouperr / 100)
                io.cprint(batch_print)
                total_grouperr = 0.0
                total_loss = 0.0
                total_score_loss = 0.0
            batch_count += 1

        model_path = os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch + 1) + '.pt')
        torch.save(model.state_dict(), model_path)
        save_print = 'Successfully store the checkpoint model into ' + model_path
        io.cprint(save_print)

if __name__ == "__main__":

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)

