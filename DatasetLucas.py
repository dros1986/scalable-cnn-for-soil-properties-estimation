import os
import random
import torch
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from decimal import *


class DatasetLucas(object):
    def __init__(self, csv, batch_size=100):
        # save batch size
        self.batch_size = batch_size
        # read csv files
        df = pd.read_table(csv, sep=',')
        # get x of ir
        self.ir_x = np.sort(np.array([float(col[4:]) for col in df.columns if 'spc.' in col]))
        # get ir
        ir_cols = ['spc.' + str(col).rstrip('0').rstrip('.') for col in self.ir_x]
        self.ir_y = df[ir_cols].to_numpy()
        # resample to match pignoletto standard
        new_ir_x = np.arange(400, 2501, step=1)
        new_ir_y = np.zeros((self.ir_y.shape[0], new_ir_x.shape[0]))
        for i in range(self.ir_y.shape[0]):
            new_ir_y[i] = np.interp(
                new_ir_x, # where to interpret
                self.ir_x,    # known positions
                self.ir_y[i], # known data points
            )
        self.ir_x = new_ir_x
        self.ir_y = new_ir_y
        # normalize signal
        self.ir_y = self.ir_y / np.expand_dims(self.ir_y.sum(1),1)
        # unsqueeze and make it torch
        self.ir_y = torch.from_numpy(self.ir_y).float().unsqueeze(1).unsqueeze(1)
        # get target variables
        tg_cols = ['pH.in.H2O', 'P', 'N', 'coarse', 'sand', 'silt', 'clay']
        self.tg_vars = df[tg_cols].to_numpy()
        self.tg_vars = torch.from_numpy(self.tg_vars).float()
        import ipdb; ipdb.set_trace()
        # define number of batches
        self.n_batches = self.tg_vars.shape[0] // self.batch_size
        # define random sequence
        self.seq = list(range(self.tg_vars.shape[0]))
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
            return self.ir_y[ids], self.tg_vars[ids]

    def __len__(self):
        return self.n_batches


    def get_ir(self, n_components=None):
        if n_components is not None and not n_components == 0:
            # remove dimensions through pca
            pca = PCA(n_components=n_components)
            x_pca = pca.fit_transform(self.ir_y)
            # create dataframe
            res = pd.DataFrame(data=x_pca, index=self.ir_x, columns=range(x_pca.shape[1]))
            return res
        else:
            return self.ir_y


    # def get_locations(self):
    #     pass

    # def joined(self, ir_n_components=None):
    #     pass


if __name__ == '__main__':
    data = DatasetLucas(batch_size=500)
    for cur_in, cur_gt in data:
        print(cur_in.shape)
        print(cur_gt.shape)
