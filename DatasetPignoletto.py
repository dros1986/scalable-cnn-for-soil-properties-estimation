import os
import pandas as pd
from sklearn.decomposition import PCA


class DatasetPignoletto(object):
    def __init__(self, dir='data'):
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


if __name__ == '__main__':
    data = Data()
    tot, src_keys, tgt_keys = data.joined()
    tot2, src_keys2, tgt_keys2 = data.joined(ir_n_components=0.99)
    import ipdb
    ipdb.set_trace()
