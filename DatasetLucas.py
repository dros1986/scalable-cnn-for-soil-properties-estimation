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
            src_norm,
            tgt_norm,
            tgt_quant,
            src_prefix = 'spc.',
            tgt_vars = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
            fmin = None,
            fmax = None,
            batch_size = 100,
            sep = ',',
            drop_last = True,
            shuffle = True,
            return_coords = False
        ):
        # save attributes
        self.src_norm = src_norm
        self.tgt_norm = tgt_norm
        self.tgt_quant = tgt_quant
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.fmin = fmin
        self.fmax = fmax
        # read csv files
        df = pd.read_table(csv, sep=sep)
        # get source and target variables
        self.src_cols, self.src_x, self.src_y = self.get_src(df, src_prefix, fmin, fmax)
        self.tgt_vars = self.get_tgt(df, tgt_vars)
        # check size
        assert(self.src_y.shape[0] == self.tgt_vars.shape[0])
        # normalize source and target variables
        self.src_y = self.src_norm(self.src_y)
        self.tgt_vars = self.tgt_norm(self.tgt_vars)
        # quantize target variables
        self.tgt_bins, self.tgt_regs = self.tgt_quant(self.tgt_vars)
        # get coordinates
        self.return_coords = return_coords
        self.coords = torch.from_numpy(df[['GPS_LAT','GPS_LONG']].to_numpy()).float()
        # define number of batches
        if drop_last:
            self.n_batches = self.tgt_vars.size(0) // self.batch_size
        else:
            self.n_batches = math.ceil(self.tgt_vars.size(0) / self.batch_size)
        # define random sequence
        self.seq = list(range(self.tgt_vars.shape[0]))
        if self.shuffle:
            random.shuffle(self.seq)
        # define current batch number
        self.cur_batch = 0


    def get_src(self, df, src_prefix, fmin, fmax):
        ''' gets infrared signals using cols starting with src_prefix '''
        # get x points
        src_cols = [col for col in df.columns if col.startswith(src_prefix)]
        x = np.array([float(col[len(src_prefix):]) for col in src_cols])
        # sort x points in increasing order
        pos = np.argsort(x)
        src_cols = [src_cols[cur_pos] for cur_pos in pos]
        # extract x and y values
        src_x = x[pos]
        src_y = df[src_cols].to_numpy()
        # keep only frequences in range
        cond = np.ones(src_x.shape).astype(np.bool)
        if fmin is not None:
            # import ipdb; ipdb.set_trace()
            cond *= src_x>=fmin
        if fmax is not None:
            cond *= src_x<=fmax
        src_y = src_y[:,cond]
        src_cols = [cur_col for (cur_col, cur_cond) in zip(src_cols, cond) if cur_cond]
        src_x = src_x[cond]
        # convert to tensor
        src_x = torch.from_numpy(src_x).float()
        src_y = torch.from_numpy(src_y).float()
        # return
        return src_cols, src_x, src_y


    def get_tgt(self, df, tgt_vars):
        ''' gets target variables using specified columns '''
        # extract variables
        tgt_vars = df[tgt_vars].to_numpy()
        # convert to torch
        tgt_vars = torch.from_numpy(tgt_vars).float()
        # return them
        return tgt_vars


    def __iter__(self):
        return self

    def __next__(self):
        if not self.cur_batch < self.n_batches:
            self.cur_batch = 0
            if self.shuffle:
                random.shuffle(self.seq)
            raise StopIteration
        else:
            isrt = self.cur_batch*self.batch_size
            iend = isrt + self.batch_size
            ids = self.seq[isrt:iend]
            self.cur_batch += 1
            src_y = self.src_y[ids].unsqueeze(1).unsqueeze(1).float()
            tgt = self.tgt_vars[ids].float()
            bins = self.tgt_bins[ids]
            regs = self.tgt_regs[ids]
            if self.return_coords:
                return src_y, tgt, bins, regs, self.coords[ids]
            return src_y, tgt, bins, regs

    def __len__(self):
        return self.n_batches


if __name__ == '__main__':
    # import only for testing
    from Normalization import InstanceStandardization, VariableStandardization
    from Quantizer import Quantizer
    # instantiate normalizer and quantizer
    src_norm = InstanceStandardization()
    tgt_norm = VariableStandardization(12)
    quant = Quantizer(nvars=12, nbins=10)
    # instantiate dataset
    csv = '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_val.csv'
    # data = DatasetLucas(csv, src_norm, tgt_norm, quant, fmin=500, fmax=1000, batch_size=500)
    data = DatasetLucas(csv, src_norm, tgt_norm, quant, batch_size=100)
    # boh = DatasetLucas(csv, src_norm, tgt_norm, quant, batch_size=500)
    print(data.src_x.min())
    print(data.src_x.max())
    print('---')
    # check
    for cur_in, cur_gt, cur_bin, cur_reg in data:
        print(cur_in.shape)
        print(cur_gt.shape)
        print(cur_bin.shape)
        print(cur_reg.shape)
        # import matplotlib.pyplot as plt
        # for i in range(cur_in.size(0)):
        #     plt.plot(data.src_x, cur_in.squeeze()[i])
        # plt.show()
        # import ipdb; ipdb.set_trace()
        break
