import os
import pandas as pd


class Data(object):
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

    def get_ir(self):
        return self.ir

    def get_gamma(self):
        return self.gamma

    def get_locations(self):
        pass

    def joined(self):
        # merge
        tot = pd.merge(self.ir, self.gamma, left_index=True, right_index=True)
        tot = pd.merge(tot, self.resistivita, left_index=True, right_index=True)
        # define source and target keys
        src_keys = list(self.ir.columns) + list(self.gamma.columns)
        tgt_keys = list(self.resistivita.columns)
        # return df with keys
        return tot, src_keys, tgt_keys


if __name__ == '__main__':
    data = Data()
    tot, src_keys, tgt_keys = data.joined()
    import ipdb; ipdb.set_trace()
