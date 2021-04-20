import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argument_parser import parse_train_arguments
from Quantizer import Quantizer
from Normalization import *
from DatasetLucas import DatasetLucas
from networks import Net
from test import test, is_better


def train(args, net, optimizer, train_dataset, loss_fn, writer, epoch, best_val):
    # set network in training mode
    net.train()
    # for each element
    for i, (src, tgt, bins, reg) in enumerate(train_dataset):
        # move in device
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        bins = bins.to(args.device)
        reg = reg.to(args.device)
        # reset grads
        optimizer.zero_grad()
        # apply network
        out = net(src)
        # project in output space
        # apply loss
        cur_l = loss_fn(out, tgt, bins, reg)
        # write on tensorboard
        writer.add_scalar('loss', cur_l, (ne*train_dataset.n_batches)+i)
        # print
        print('[{:d}/{:d}] Loss: {:.4f} - Val: {:.4f}' \
            .format(
            epoch, args.nepochs, cur_l,
            best_val[args.val].loc['global'].value if best_val is not None else 0.0))
        # backward propagation
        cur_l.backward()
        # update weights
        optimizer.step()


def get_metric_and_loss(loss):
    if loss == 'l1':
        loss_fun = lambda x,y,b,r: F.l1_loss(x,y)
        metric_fun = nn.Identity()
    elif loss == 'l2' or loss == 'mse':
        loss_fun = lambda x,y,b,r: F.mse_loss(x,y)
        metric_fun = nn.Identity()
    elif loss == 'classification':
        loss_fun = 0
        def metric_fun(x,y,b,r):
            pass
    return metric_fun, loss_fun


def save_checkpoint(net, optimizer, tgt_norm, tgt_quant, cur_val, best_val, epoch, expdir, is_best=False):
    state = {   'net':net.state_dict(),
                'opt':optimizer.state_dict(),
                'tgt_norm': tgt_norm.get_state(),
                'tgt_quant': tgt_quant.get_state(),
                'cur_val':cur_val,
                'best_val': best_val,
                'epoch':ne}
    torch.save(state, os.path.join(expdir, 'checkpoints','{:d}.pth'.format(ne)))
    if is_best:
        torch.save(state, os.path.join(expdir, 'checkpoints','best.pth'))


if __name__ == '__main__':
    # parse arguments
    args = parse_train_arguments()
    # create output dir
    os.makedirs(args.experiment, exist_ok=True)
    os.makedirs(os.path.join(args.experiment, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.experiment, 'logs'), exist_ok=True)
    # define network
    net = Net(nemb=12, nch=1, powf=args.powf, max_powf=args.max_powf, insz=args.insz, \
            minsz=args.minsz, nbsr=args.nbsr, leak=args.leak, batch_momentum=args.momentum)
    net.to(args.device)
    # init optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, \
                        weight_decay=args.weight_decay)
    # init loss and metric
    metric_fun, loss_fun = get_metric_and_loss(args.loss)
    # load from checkpoint if required
    if args.start_from >= 0:
        state = os.path.join(args.experiment,'{:d}.pth'.format(args.start_from))
        net.load_state_dict(state['net'])
        opt.load_state_dict(state['opt'])
        best_val = state['best_val']
        start_epoch = state['epoch'] + 1
    else:
        best_val = None
        start_epoch = 0
    # create normalization objects
    src_norm = InstanceStandardization()
    tgt_norm = VariableStandardization()
    # define quantization object
    tgt_quant = Quantizer(nbins=args.nbins)
    # define datasets
    train_dataset = DatasetLucas(args.train_csv, src_norm, tgt_norm, tgt_quant, \
                            src_prefix=args.src_prefix, tgt_vars=args.tgt_vars, \
                            batch_size=args.batchsize, drop_last=True)
    valid_dataset = DatasetLucas(args.train_csv, src_norm, tgt_norm, tgt_quant, \
                            src_prefix=args.src_prefix, tgt_vars=args.tgt_vars, \
                            batch_size=args.batchsize, drop_last=False)
    # init tensorboard writer
    writer = SummaryWriter(os.path.join(args.experiment,'logs'))
    # write args in tensorboard
    arg_dict = vars(args)
    text = ''.join([key + ': ' + str(arg_dict[key]) + '  \n' for key in arg_dict])
    writer.add_text('parameters', text, global_step=0)
    # for each epoch
    for ne in range(start_epoch, args.nepochs):
        train(args, net, optimizer, train_dataset, loss_fun, writer, ne, best_val)
        cur_val = test(net, src_norm, tgt_norm, tgt_quant, valid_dataset, args.tgt_vars, device='cuda')
        # print current value
        print(cur_val[args.val])
        # save on tensorboard
        for cur_metric in cur_val:
            for cur_var in cur_val[cur_metric].index:
                writer.add_scalar(cur_metric + '/' + cur_var, cur_val[cur_metric].loc[cur_var].value, ne)
        # check if it's the best
        if is_better(cur_val, best_val, args.val):
            best_val = cur_val
            save_checkpoint(net, optimizer, tgt_norm, tgt_quant, cur_val, best_val, ne, args.experiment, is_best=True)
            writer.add_scalar(args.val + '/best', cur_val[args.val].loc['global'].value, ne)
        else:
            save_checkpoint(net, optimizer, tgt_norm, tgt_quant, cur_val, best_val, ne, args.experiment, is_best=False)
