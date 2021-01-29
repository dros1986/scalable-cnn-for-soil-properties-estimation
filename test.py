import os
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DatasetLucas import DatasetLucas
from networks import Net

# define test procedure
def test(net, ds):
    net.eval()
    loss = None
    for src,tgt in ds:
        # move in device
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        # compute prediction
        with torch.no_grad():
            out = net(src)
        # compare
        cur_l = F.l1_loss(out,tgt,reduction='none').cpu()
        loss = cur_l if loss is None else torch.cat((loss,cur_l), axis=0)
    # compute average
    var_loss = loss.mean(0)
    tot_loss = loss.mean()
    # create dataframe
    var_names = ds.tgt_names
    df = pd.DataFrame(columns=['variable', 'value'])
    # fill dataframe
    for (vn,vl) in zip(var_names, var_loss):
        df = df.append({'variable': vn, 'value': vl.item()}, ignore_index=True)
    df = df.append({'variable': 'avg', 'value': tot_loss.item()}, ignore_index=True)
    df = df.round(4)
    df.set_index('variable', inplace=True)
    # net back to train
    net.train()
    return df



if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batchsize", help="Batch size.",
    					default=100, type=int)
    parser.add_argument("-dev", "--device", help="Device.",
    					default='cpu', type=str)
    # network
    parser.add_argument("-powf", "--powf", help="Power of 2 of filters.",
    					default=3, type=int)
    parser.add_argument("-max_powf", "--max_powf", help="Max power of 2 of filters.",
    					default=8, type=int)
    parser.add_argument("-insz", "--insz", help="Input size.",
    					default=1024, type=int)
    parser.add_argument("-minsz", "--minsz", help="Min size.",
    					default=8, type=int)
    parser.add_argument("-nbsr", "--nbsr", help="Number of blocks same resolution.",
    					default=1, type=int)
    parser.add_argument("-leak", "--leak", help="Leak of relu. If 0, normal relu.",
    					default=0, type=float)
    parser.add_argument("-mom", "--momentum", help="Batchnorm momentum.",
    					default=0.01, type=float)
    # other params
    parser.add_argument("-exp", "--experiment", help="Name of experiment.",
    					default='experiment1', type=str)
    parser.add_argument('-last', '--latest', action='store_true')
    parser.add_argument("-norm", "--norm_type", help="Normalization used in nir/swir.",
                        default='std_instance', type=str)
    parser.add_argument("-csv", "--csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_val.csv')
    args = parser.parse_args()
    # create output dir
    os.makedirs(args.experiment, exist_ok=True)
    # define network
    net = Net(nemb=12, nch=1, powf=args.powf, max_powf=args.max_powf, insz=args.insz, minsz=args.minsz, \
            nbsr=args.nbsr, leak=args.leak, batch_momentum=args.momentum)
    net.to(args.device)
    net.eval()
    # load weights and vars
    wfn = os.path.join(args.experiment, 'latest.pth' if args.latest else 'best.pth')
    state = torch.load(wfn)
    net.load_state_dict(state['net'])
    vars = state['vars']
    # define datasets
    test_dataset = DatasetLucas(csv = args.csv, batch_size = args.batchsize, norm_type = args.norm_type, \
                                drop_last=False, vars=vars)
    # test on test dataset
    df = test(net, test_dataset)
    print('\n',df,'\n')
    # save to csv
    df.to_csv(os.path.join(args.experiment, 'results.csv'))
    # save version to file
    version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode("utf-8")
    f = open(os.path.join(args.experiment, 'version.txt'), 'w')
    f.write(version)
    f.close()
