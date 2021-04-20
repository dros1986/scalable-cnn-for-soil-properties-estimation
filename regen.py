import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argument_parser import parse_test_arguments
from Quantizer import Quantizer
from Normalization import *
from DatasetLucas import DatasetLucas
from networks import Net


def regen(net, src_norm, tgt_norm, tgt_quant, test_dataset, tgt_vars, device='cuda'):
    # set network to test
    net.eval()
    # create empty dataframe
    dfout = None
    dftgt = None
    # estimate
    for src,tgt,bin,reg in test_dataset:
        # move in device
        src = src.to(device)
        tgt = tgt.to(device)
        # compute prediction
        with torch.no_grad():
            out = net(src)
        # reverse normalization and quantization
        tgt = tgt_norm.invert(tgt.cpu())
        out = tgt_norm.invert(out.cpu())
        # out = tgt_quant.unquantize(out.cpu())
        # append
        dfout = out if dfout is None else torch.cat((dfout, out), 0)
        dftgt = tgt if dftgt is None else torch.cat((dftgt, tgt), 0)
    # make dataframes
    dfout = pd.DataFrame(data=dfout.numpy(), columns=tgt_vars)
    dftgt = pd.DataFrame(data=dftgt.numpy(), columns=tgt_vars)
    # return the dataframes
    return dfout, dftgt



if __name__ == '__main__':
    # parse arguments
    args = parse_test_arguments()
    # define network
    net = Net(nemb=12, nch=1, powf=args.powf, max_powf=args.max_powf, insz=args.insz, \
            minsz=args.minsz, nbsr=args.nbsr, leak=args.leak, batch_momentum=args.momentum)
    net.to(args.device)
    # create normalization objects
    src_norm = InstanceStandardization()
    tgt_norm = VariableStandardization()
    # define quantization object
    tgt_quant = Quantizer(nbins=args.nbins)
    # init loss and metric
    # metric_fun, loss_fun = get_metric_and_loss(args.loss)
    # load from checkpoint if required
    state = torch.load(os.path.join(args.experiment,'checkpoints','best.pth'))
    net.load_state_dict(state['net'])
    best_val = state['best_val']
    # src_norm.set_state(state['src_norm'])
    tgt_norm.set_state(state['tgt_norm'])
    tgt_quant.set_state(state['tgt_quant'])
    # set net in eval mode
    net.eval()
    # define dataset
    test_dataset = DatasetLucas(args.test_csv, src_norm, tgt_norm, tgt_quant, \
                            src_prefix=args.src_prefix, tgt_vars=args.tgt_vars, \
                            batch_size=args.batchsize, drop_last=False, shuffle=False)
    # regen
    dfout, dftgt = regen(net, src_norm, tgt_norm, tgt_quant, test_dataset, args.tgt_vars, device=args.device)
    # create output csv
    out_csv_fn = os.path.join(args.experiment,'best.csv')
    # save
    dfout.round(decimals=4).to_csv(out_csv_fn, sep=';', index=False)
