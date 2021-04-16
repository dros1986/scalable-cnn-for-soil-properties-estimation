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


def test(args, net, val_dataset):
    net.eval()
    loss = None
    for src,tgt,bin,reg in val_dataset:
        # move in device
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        # compute prediction
        with torch.no_grad():
            out = net(src)
        # compare
        cur_l = F.mse_loss(out,tgt,reduction='none').cpu()
        loss = cur_l if loss is None else torch.cat((loss,cur_l), axis=0)
    # compute average
    var_loss = torch.sqrt(loss.mean(0))
    tot_loss = torch.sqrt(loss.mean())
    # fill DataFrame
    var_names = args.tgt_vars
    df = pd.DataFrame(columns=['variable', 'value'])
    # put variables inside dataframe
    for (vn,vl) in zip(var_names, var_loss):
        df = df.append({'variable': vn, 'value': vl.item()}, ignore_index=True)
    # sort by value
    df.sort_values('value', ascending=True, inplace=True)
    # append average
    df = df.append({'variable': 'avg', 'value': tot_loss.item()}, ignore_index=True)
    df = df.round(4)
    df.set_index('variable', inplace=True)
    # net back to train
    net.train()
    return df, tot_loss.item()


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
            best_val.loc['avg'].value if best_val is not None else 0.0))
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


if __name__ == '__main__':
    # parse arguments
    args = parse_train_arguments()
    # create output dir
    os.makedirs(args.experiment, exist_ok=True)
    # define network
    net = Net(nemb=12, nch=1, powf=args.powf, max_powf=args.max_powf, insz=args.insz, \
            minsz=args.minsz, nbsr=args.nbsr, leak=args.leak, batch_momentum=args.momentum)
    net.to(args.device)
    # init optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, \
                        weight_decay=args.weight_decay)
    # init loss and metric
    metric_fun, loss_fun = get_metric_and_loss(args.loss)
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
    writer = SummaryWriter(args.experiment)
    # write args in tensorboard
    arg_dict = vars(args)
    text = ''.join([key + ': ' + str(arg_dict[key]) + '  \n' for key in arg_dict])
    writer.add_text('parameters', text, global_step=0)
    # init metric
    best_val = None
    # for each epoch
    for ne in range(args.nepochs):
        train(args, net, optimizer, train_dataset, loss_fun, writer, ne, best_val)
        # test on validation dataset
        df, loss = test(args, net, valid_dataset)
        print('\n',df,'\n')
        # save on tensorboard
        for col in df.index:
            writer.add_scalar(col, df.loc[col].value, ne)
        # create state
        state = {   'net':net.state_dict(),
                    'opt':optimizer.state_dict(),
                    'tgt_norm': tgt_norm.get_state(),
                    'tgt_quant': tgt_quant.get_state(),
                    'res':df}
        torch.save(state, os.path.join(args.experiment,'{:d}.pth'.format(ne)))
        # check if it is the best
        if best_val is None or df.loc['avg'].value < best_val.loc['avg'].value:
            best_val = df
            writer.add_scalar('best', best_val.loc['avg'].value, ne)
            torch.save(state, os.path.join(args.experiment,'best.pth'))
