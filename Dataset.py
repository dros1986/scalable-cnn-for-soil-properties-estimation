import os
import math
import random
import torch
import pandas as pd
import numpy as np
from scipy import signal
from Normalization import InstanceStandardization, VariableStandardization


class Dataset(object):
    def __init__(self,
            csv,
            # signals
            hyper_prefix = 'spc.',
            multi_prefix = 'multi.',
            hyper_fmin = None,
            hyper_fmax = None,
            multi_fmin = None,
            multi_fmax = None,
            # variables
            chemicals = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
            geomorpho = ['height','valley'],
            # normalization
            hyper_norm = InstanceStandardization(),
            multi_norm = InstanceStandardization(),
            chemicals_norm = VariableStandardization(12),
            geomorpho_norm = VariableStandardization(2),
            # coordinates
            lat_key = 'GPS_LAT',
            lon_key = 'GPS_LONG',
            # other
            batch_size = 100,
            sep = ',',
            drop_last = True,
            shuffle = True,
            return_coords = False
        ):
        # read csv file
        df = pd.read_table(csv, sep=sep)
        # get signals
        self.hyper_cols, self.hyper_bands, self.hyper_vals = \
                        self.get_signal(df, hyper_prefix, hyper_fmin, hyper_fmax)
        self.multi_cols, self.multi_bands, self.multi_vals = \
                        self.get_signal(df, multi_prefix, multi_fmin, multi_fmax)
        # get variables
        self.chemicals = self.get_vars(df, chemicals)
        self.geomorpho = self.get_vars(df, geomorpho)
        # check sizes
        self.check()
        # normalize
        self.hyper_vals = hyper_norm(self.hyper_vals)
        self.multi_vals = multi_norm(self.multi_vals)
        self.chemicals = chemicals_norm(self.chemicals)
        self.geomorpho = geomorpho_norm(self.geomorpho)
        # get coordinates
        self.return_coords = return_coords
        self.coords = torch.from_numpy(df[[lat_key,lon_key]].to_numpy()).float()
        # save attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        # define number of batches
        if drop_last:
            self.n_batches = self.chemicals.size(0) // self.batch_size
        else:
            self.n_batches = math.ceil(self.chemicals.size(0) / self.batch_size)
        # define random sequence
        self.seq = list(range(self.chemicals.size(0)))
        if self.shuffle:
            random.shuffle(self.seq)
        # define current batch number
        self.cur_batch = 0


    def get_signal(self, df, prefix, fmin=None, fmax=None):
        '''
        Gets signals using cols starting with hyper_prefix.
        The suffix must be proceded by the wavelength value. E.G.: spc.400
        '''
        # get x points
        src_cols = [col for col in df.columns if col.startswith(prefix)]
        x = np.array([float(col[len(prefix):]) for col in src_cols])
        # sort x points in increasing order
        pos = np.argsort(x)
        src_cols = [src_cols[cur_pos] for cur_pos in pos]
        # extract x and y values
        src_x = x[pos]
        src_y = df[src_cols].to_numpy()
        # keep only frequences in range fmin-fmax if specified
        cond = np.ones(src_x.shape).astype(np.bool)
        if fmin is not None:
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


    def get_vars(self, df, var_names):
        ''' gets variables specified in var_names '''
        # extract variables
        tgt_vars = df[var_names].to_numpy()
        # convert to torch
        tgt_vars = torch.from_numpy(tgt_vars).float()
        # return them
        return tgt_vars


    def check(self):
        assert(self.hyper_vals.shape[0] == self.multi_vals.shape[0])
        assert(self.hyper_vals.shape[0] == self.chemicals.shape[0])
        assert(self.hyper_vals.shape[0] == self.geomorpho.shape[0])


    def __iter__(self):
        return self

    def __next__(self):
        # if last batch, shuffle and start from beginning
        if not self.cur_batch < self.n_batches:
            self.cur_batch = 0
            if self.shuffle:
                random.shuffle(self.seq)
            raise StopIteration
        else:
            # define start and end index of current batch
            isrt = self.cur_batch*self.batch_size
            iend = isrt + self.batch_size
            # get the ids of samples in current batch
            ids = self.seq[isrt:iend]
            # get variables
            hyper = self.hyper_vals[ids].unsqueeze(1).float()
            multi = self.multi_vals[ids].unsqueeze(1).float()
            chemi = self.chemicals[ids].float()
            geomo = self.geomorpho[ids].float()
            # increase batch number
            self.cur_batch += 1
            # include coordinates if needed and return
            if self.return_coords:
                return hyper, multi, chemi, geomo, self.coords[ids]
            return hyper, multi, chemi, geomo

    def __len__(self):
        return self.n_batches


if __name__ == '__main__':
    # define csv path
    # csv = '/home/flavio/datasets/LucasLibrary/shared/lucas_dataset_val.csv'
    csv = 'fake.csv'
    # init dataset
    data = Dataset(csv, batch_size=100)
    # print min and max of hyper
    print(data.hyper_bands.min())
    print(data.hyper_bands.max())
    print('---')
    # check
    for hyper, multi, chemi, geomo in data:
        print(hyper.shape)
        print(multi.shape)
        print(chemi.shape)
        print(geomo.shape)
        print('---')
        # import matplotlib.pyplot as plt
        # for i in range(cur_in.size(0)):
        #     plt.plot(data.src_x, cur_in.squeeze()[i])
        # plt.show()
        # import ipdb; ipdb.set_trace()
        # break
