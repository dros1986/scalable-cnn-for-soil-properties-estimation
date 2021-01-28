import os
import math
import random
import torch
import pandas as pd
import numpy as np
from scipy import signal


class DatasetLucas(object):
    def __init__(self,
            csv,
            src_prefix='spc.',
            tgt_vars=['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
            batch_size=100,
            norm_type = 'std_instance',
            sep=',',
            drop_last = True,
            vars = None
        ):
        # save batch size
        self.batch_size = batch_size
        # save vars
        self.vars = vars if vars is not None else {}
        # read csv files
        df = pd.read_table(csv, sep=sep)
        # get source variables
        src_cols = [col for col in df.columns if col.startswith(src_prefix)]
        x = np.array([float(col[len(src_prefix):]) for col in src_cols])
        # sort in increasing order
        pos = np.argsort(x)
        self.src_cols = [src_cols[cur_pos] for cur_pos in pos]
        self.src_x = x[pos]
        self.src_y = df[self.src_cols]
        self.src_x = torch.from_numpy(self.src_x)
        self.src_y = torch.from_numpy(self.src_y.to_numpy())
        # standardize nir/swir
        self.std_inplace(norm_type, data_attr='src_y', mu_key='src_y_mu', vr_key='src_y_vr')
        # get target vars
        self.tgt_names = tgt_vars
        self.vars['tgt_names'] = self.tgt_names
        self.tgt_vars = df[tgt_vars].to_numpy()
        self.tgt_vars = torch.from_numpy(self.tgt_vars)
        # standardize variables independently
        self.std_inplace('std_var', data_attr='tgt_vars', mu_key='tgt_vars_mu', vr_key='tgt_vars_vr')
        # define number of batches
        if drop_last:
            self.n_batches = self.tgt_vars.size(0) // self.batch_size
        else:
            self.n_batches = math.ceil(self.tgt_vars.size(0) / self.batch_size)
        # define random sequence
        self.seq = list(range(self.tgt_vars.shape[0]))
        random.shuffle(self.seq)
        # define current batch number
        self.cur_batch = 0


    def std_inplace(self, std_type, data_attr='src_ir', mu_key='src_y_mu', vr_key='src_y_vr'):
        # retrieve mean and variance
        if mu_key in self.vars and vr_key in self.vars:
            mu = self.vars[mu_key]
            vr = self.vars[vr_key]
        else:
            mu = None
            vr = None
        # compute standardization
        feats = eval('self.'+data_attr)
        std_func = eval('self.'+std_type)
        feats, mu, vr = std_func(feats, mu, vr)
        # reassign values inplace
        exec('self.{} = feats'.format(data_attr))
        self.vars[mu_key] = mu
        self.vars[vr_key] = vr


    def std_instance(self, feats, mu = None, vr = None):
        mu = feats.mean(1).unsqueeze(1)
        vr = feats.var(1).unsqueeze(1)
        return self.std_formula(feats, mu, vr)


    def std_global(self, feats, mu = None, vr = None):
        if mu == None or vr == None:
            mu = feats.mean()
            vr = feats.var()
        return self.std_formula(feats, mu, vr)


    def std_var(self, feats, mu = None, vr = None):
        if mu == None or vr == None:
            mu = feats.mean(0).unsqueeze(0)
            vr = feats.var(0).unsqueeze(0)
        return self.std_formula(feats, mu, vr)


    def std_formula(self, feats, mu, vr):
        feats = (feats - mu) / vr
        return feats, mu, vr

    def get_vars(self):
        return self.vars

    def __iter__(self):
        return self

    def __next__(self):
        if not self.cur_batch < self.n_batches:
            self.cur_batch = 0
            random.shuffle(self.seq)
            raise StopIteration
        else:
            isrt = self.cur_batch*self.batch_size
            iend = isrt + self.batch_size
            ids = self.seq[isrt:iend]
            self.cur_batch += 1
            src_y = self.src_y[ids].unsqueeze(1).unsqueeze(1).float()
            tgt = self.tgt_vars[ids].float()
            return src_y, tgt

    def __len__(self):
        return self.n_batches


if __name__ == '__main__':
    csv = 'data/LUCAS.SOIL_corr_FULL.csv'
    data = DatasetLucas(csv, batch_size=500)
    for cur_in, cur_gt in data:
        print(cur_in.shape)
        print(cur_gt.shape)
