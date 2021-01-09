import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


class DatasetPignoletto(object):
    def __init__(self, dir='data', batch_size=25):
        # read csv files
        ir = pd.read_csv(os.path.join(dir, 'ir.csv'), sep=';')
        map = pd.read_csv(os.path.join(dir, 'map.csv'), sep=';')
        per_informatici = pd.read_csv(os.path.join(dir, 'per_informatici.csv'), sep=';')
        resistivita = pd.read_csv(os.path.join(dir, 'resistivita.csv'), sep=';')
        # inner join map - resistivita
        resistivita = pd.merge(map.set_index('campo'), resistivita,
                               left_on='campo', right_on='Punto')
        # set lab as index
        resistivita = resistivita.set_index('lab')
        # remove 'res_Text', 'prof cm' and 'Punto' from resistivita
        self.resistivita = resistivita.drop(['Text', 'prof cm', 'Punto'], axis=1)
        # get nir-swir
        self.ir = ir.set_index('sample')
        # get gamma
        self.gamma = per_informatici.set_index('SAMPLE')
        # save mapping
        self.map = map
        # get source for iterator
        self.ir_x = np.sort(np.array([float(x) for x in self.ir.columns]))
        self.ir_y = self.ir[[str(int(cur_b)) for cur_b in self.ir_x]].to_numpy()
        # normalize ir signal
        self.ir_y = self.ir_y / np.expand_dims(self.ir_y.sum(1),1)
        # unsqueeze and make it torch
        self.ir_y = torch.from_numpy(self.ir_y).float().unsqueeze(1).unsqueeze(1)
        # get target
        tg_cols = ['pH', 'P (mg/kg)', 'N (%)', 'Sg1 (g/kg)', 'St (g/kg)', 'L (g/kg)', 'A (g/kg)']
        self.tg_vars = resistivita[tg_cols].to_numpy()
        self.tg_vars = torch.from_numpy(self.tg_vars).float()
        # create batch info
        self.batch_size = batch_size
        self.n_batches = self.ir_y.shape[0] // self.batch_size
        self.cur_batch = 0
        # define random sequence
        self.seq = list(range(self.tg_vars.shape[0]))
        random.shuffle(self.seq)


    def get_resistivita(self):
        return self.resistivita

    def get_ir(self, n_components=None):
        if n_components is not None and not n_components == 0:
            # remove dimensions through pca
            pca = PCA(n_components=n_components)
            x_pca = pca.fit_transform(self.ir.to_numpy())
            # create dataframe
            res = pd.DataFrame(data=x_pca, index=self.ir.index, columns=range(x_pca.shape[1]))
            return res
        else:
            return self.ir

    def get_gamma(self):
        return self.gamma

    def get_locations(self):
        pass

    def joined(self, ir_n_components=None):
        # get nir-swir
        ir = self.get_ir(n_components=ir_n_components)
        # merge
        tot = pd.merge(ir, self.gamma, left_index=True, right_index=True)
        tot = pd.merge(tot, self.resistivita, left_index=True, right_index=True)
        # define source and target keys
        src_keys = list(ir.columns) + list(self.gamma.columns)
        tgt_keys = list(self.resistivita.columns)
        # return df with keys
        return tot, src_keys, tgt_keys

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


if __name__ == '__main__':
    data = DatasetPignoletto()
    tot, src_keys, tgt_keys = data.joined()
    tot2, src_keys2, tgt_keys2 = data.joined(ir_n_components=0.99)
    # test iter
    for src,tgt in data:
        print(src.shape)
        print(tgt.shape)
        print('----')
    # import ipdb
    # ipdb.set_trace()
