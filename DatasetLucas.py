import os
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
            sep=','
        ):
        # save batch size
        self.batch_size = batch_size
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
        # standardize all bands together
        self.src_y_mu = self.src_y.mean()
        self.src_y_vr = self.src_y.var()
        self.src_y = (self.src_y - self.src_y_mu) / self.src_y_vr
        # get target vars
        self.tgt_vars = df[tgt_vars].to_numpy()
        self.tgt_vars = torch.from_numpy(self.tgt_vars)
        # standardize variables independently
        self.tgt_vars_mu = self.tgt_vars.mean(0).unsqueeze(0)
        self.tgt_vars_vr = self.tgt_vars.var(0).unsqueeze(0)
        self.tgt_vars = (self.tgt_vars - self.tgt_vars_mu) / self.tgt_vars_vr
        # define number of batches
        self.n_batches = self.tgt_vars.size(0) // self.batch_size
        # define random sequence
        self.seq = list(range(self.tgt_vars.shape[0]))
        random.shuffle(self.seq)
        # define current batch number
        self.cur_batch = 0


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
            return self.src_y[ids], self.tgt_vars[ids]

    def __len__(self):
        return self.n_batches


if __name__ == '__main__':
    csv = '/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL.csv'
    data = DatasetLucas(csv, batch_size=500)
    for cur_in, cur_gt in data:
        print(cur_in.shape)
        print(cur_gt.shape)
