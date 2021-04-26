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
from test import test_batch #, is_better

import pytorch_lightning as pl


class Experiment(pl.LightningModule):
    def __init__(self, conf):
        super(Experiment, self).__init__()
        # save parameters
        # self.save_hyperparameters(conf) # not working
        self.hparams = conf
        # define network
        self.net = Net(nemb=12, nch=1, powf=conf['powf'], max_powf=conf['max_powf'], insz=conf['insz'], \
                minsz=conf['minsz'], nbsr=conf['nsbr'], leak=conf['leak'], batch_momentum=conf['batch_momentum'])
        # define metric
        self.metric_fun, self.loss_fun = self.get_metric_and_loss(conf['loss'])
        # create normalization objects
        self.src_norm = InstanceStandardization()
        self.tgt_norm = VariableStandardization()
        # define quantization object
        self.tgt_quant = Quantizer(nbins=conf['nbins'])
        # save data params
        self.train_csv = conf['train_csv']
        self.val_csv = conf['val_csv']
        self.test_csv = conf['test_csv']
        self.src_prefix = conf['src_prefix']
        self.batch_size = conf['batch_size']
        self.num_workers = conf['num_workers']
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
                                batch_size=self.batch_size, drop_last=True)


    def val_dataloader(self):
        return DatasetLucas(self.val_csv, self.src_norm, self.tgt_norm, self.tgt_quant, \
                                src_prefix=self.src_prefix, tgt_vars=self.tgt_vars, \
                                batch_size=self.batch_size, drop_last=False)


    def test_dataloader(self):
        return DatasetLucas(self.test_csv, self.src_norm, self.tgt_norm, self.tgt_quant, \
                                src_prefix=self.src_prefix, tgt_vars=self.tgt_vars, \
                                batch_size=self.batch_size, drop_last=False)



    def get_metric_and_loss(self, loss):
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
