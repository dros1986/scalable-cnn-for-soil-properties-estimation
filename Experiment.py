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

from Quantizer import Quantizer
from Normalization import *
from DatasetLucas import DatasetLucas
from networks import Net
from test import test_batch #, is_better

import pytorch_lightning as pl


class Experiment(pl.LightningModule):
    def __init__(self, conf):
        super(Experiment, self).__init__()
        # save parameters
        # self.save_hyperparameters(conf) # not working
        self.hparams = conf
        self.conf = conf
        # define output size
        if conf['loss'] == 'classification':
            outsz = len(conf['tgt_vars'])*conf['nbins'] + len(conf['tgt_vars'])
            sigmoid_from = len(conf['tgt_vars'])*conf['nbins']
        else:
            outsz = len(conf['tgt_vars'])
            sigmoid_from = None
        # define network
        self.net = Net(nemb=outsz, nch=1, powf=conf['powf'], max_powf=conf['max_powf'], insz=conf['insz'], \
                minsz=conf['minsz'], nbsr=conf['nsbr'], leak=conf['leak'], batch_momentum=conf['batch_momentum'], \
                use_batchnorm=conf['use_batchnorm'], sigmoid_from=sigmoid_from)
        # define metric
        self.loss_fun = self.get_loss(conf['loss'], conf['tgt_vars'], conf['nbins'])
        # create normalization objects
        self.src_norm = InstanceStandardization()
        self.tgt_norm = VariableStandardization(len(conf['tgt_vars']))
        # define quantization object
        self.tgt_quant = Quantizer(nbins=conf['nbins'], nvars=len(conf['tgt_vars']))
        # save data params
        self.train_csv = conf['train_csv']
        self.val_csv = conf['val_csv']
        self.test_csv = conf['test_csv']
        self.src_prefix = conf['src_prefix']
        self.batch_size = conf['batch_size']
        self.num_workers = conf['num_workers']
        self.fmin = conf['fmin'] if 'fmin' in conf else None
        self.fmax = conf['fmax'] if 'fmax' in conf else None
        # save optimizer parameters
        self.learning_rate = conf['learning_rate']
        self.weight_decay = conf['weight_decay']
        # save other params
        # self.loss = loss
        self.val = conf['val']
        self.tgt_vars = conf['tgt_vars']


    def forward(self, x):
        return self.net(x)


    def training_step(self, batch, batch_nb):
        # split components
        src, tgt, bins, reg = batch
        # get prediction
        out = self(src)
        # TODO: project in output space
        # apply loss
        loss = self.loss_fun(out, tgt, bins, reg)
        # return loss value
        self.log("train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        # split components
        src, tgt, bins, reg = batch
        # compute prediction
        with torch.no_grad():
            out = self(src)
        # project to output space
        out = self.project_to_output_space(out)
        # reverse normalization and quantization
        tgt = self.tgt_norm.invert(tgt) #.cpu())
        out = self.tgt_norm.invert(out) #.cpu())
        # test
        cur_val = test_batch(out, tgt, self.tgt_vars)
        # define output dict
        out_metrics = {}
        # for each metric
        for cur_metric in cur_val:
            for cur_var in cur_val[cur_metric].index:
                # get name and value
                metric_name = cur_metric + '/' + cur_var
                metric_val = cur_val[cur_metric].loc[cur_var].value
                # log on tb
                self.log(metric_name, metric_val, prog_bar=True)
                # append to output
                out_metrics[metric_name] = metric_val
        # return
        return out_metrics


    def validation_epoch_end(self, outputs):
        # get metrics
        metrics = outputs[0].keys()
        # for each metric, compute average over the validation steps
        for cur_metric in metrics:
            cur_avg = torch.tensor([x[cur_metric] for x in outputs]).mean()
            self.log('avg/'+cur_metric, cur_avg)


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(),
                            weight_decay = self.weight_decay,
                            lr = self.learning_rate)


    def train_dataloader(self):
        return DatasetLucas(self.train_csv, self.src_norm, self.tgt_norm, self.tgt_quant, \
                                src_prefix=self.src_prefix, tgt_vars=self.tgt_vars, \
                                fmin=self.fmin, fmax=self.fmax, \
                                batch_size=self.batch_size, drop_last=True)


    def val_dataloader(self):
        return DatasetLucas(self.val_csv, self.src_norm, self.tgt_norm, self.tgt_quant, \
                                src_prefix=self.src_prefix, tgt_vars=self.tgt_vars, \
                                fmin=self.fmin, fmax=self.fmax, \
                                batch_size=self.batch_size, drop_last=False)


    def test_dataloader(self, return_coords=False):
        return DatasetLucas(self.test_csv, self.src_norm, self.tgt_norm, self.tgt_quant, \
                                src_prefix=self.src_prefix, tgt_vars=self.tgt_vars, \
                                fmin=self.fmin, fmax=self.fmax, \
                                batch_size=self.batch_size, drop_last=False, \
                                return_coords=return_coords)



    def get_loss(self, loss, tgt_vars, nbins):
        if loss == 'l1':
            loss_fun = lambda x,y,b,r: F.l1_loss(x,y)
        elif loss == 'l2' or loss == 'mse':
            loss_fun = lambda x,y,b,r: F.mse_loss(x,y)
        elif loss == 'classification':
            loss_fun = 0
            def loss_fun(x,y,b,r):
                # nll on bins
                cl_loss = 0
                for nv in range(len(tgt_vars)):
                    cl_loss += F.nll_loss(x[:, nv*nbins:(nv+1)*nbins], b[:, nv])
                # average cl loss
                cl_loss /= len(tgt_vars)
                # define offset for regression bins
                off = nbins*len(tgt_vars)
                # l1 regression on reg
                rg_loss = F.l1_loss(x[:, off:], r)
                # return average
                return .5*cl_loss + .5*rg_loss
        # return loss
        return loss_fun


    def project_to_output_space(self, x):
        # if plain regression, return output
        if not self.conf['loss'] == 'classification':
            return x
        # get vars
        tgt_vars = self.tgt_vars
        nbins = self.conf['nbins']
        off = nbins*len(tgt_vars)
        # else reconstruct bins
        bins = torch.zeros(x.size(0),len(tgt_vars)).to(x.device)
        for nv in range(len(tgt_vars)):
            # reconstruct bins
            cur_bin = x[:, nv*nbins:(nv+1)*nbins]
            cur_bin = cur_bin.max(1)[1]
            bins[:,nv] = cur_bin
        # reconstruct regs
        reg = x[:, off:]
        # unquantize
        return self.tgt_quant.invert(bins, reg)


    def regen(self, out_fn, dl=None, sep=';'):
        # get dataloader if not specified
        if dl == None:
            dl = self.test_dataloader(return_coords=True)
        # open output file
        f = open(out_fn, 'w')
        # write header
        f.write(sep.join(['lat','lon'] + self.conf['tgt_vars']) + '\n')
        # for each batch
        for src, tgt, bins, reg, coords in dl:
            # compute prediction
            with torch.no_grad(): out = model(src)
            # project to output space
            out = self.project_to_output_space(out)
            out = self.tgt_norm.invert(out)
            # concatenate coordinates
            out = torch.cat((coords, out.cpu()),1).numpy()
            # save to file
            np.savetxt(f, out, delimiter=sep,  fmt='%.6f')
        f.close()



if __name__ == '__main__':
    # parse arguments
    # get arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-ckp", "--checkpoint", help="Checkpoint.",
    					default='', type=str)
    parser.add_argument("-out", "--out", help="Output filename.",
    					default='', type=str)
    # parse args
    args = parser.parse_args()
    # load model
    model = Experiment.load_from_checkpoint(args.checkpoint)
    model.eval()
    # if out not specified, save in folder
    if args.out == '':
        args.out = os.path.join(os.path.dirname(args.checkpoint), 'test.csv')
    # regen
    model.regen(args.out)
