import ipdb
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from DatasetPignoletto import DatasetPignoletto


def apply_pca(df, n_components=0.99):
    if n_components == 0:
        return df
    # remove dimensions through pca
    pca = PCA(n_components=n_components)
    # pca = PCA(n_components=1)
    x_pca = pca.fit_transform(df.to_numpy())
    x_red = pca.inverse_transform(x_pca)
    # create dataframe
    res = pd.DataFrame(data=x_red, index=df.index, columns=df.columns)
    return res


if __name__ == '__main__':
    # define output dir
    out_dir = 'ir_signals'
    os.makedirs(out_dir, exist_ok=True)
    # set seaborn
    sns.set(rc={'figure.figsize': (20, 10)})
    sns.set(font_scale=1.3)
    # set figures parameters
    dpi = 100
    # get data
    data = DatasetPignoletto()
    # get nir-swir
    ir = data.get_ir()
    # define pca retained variances
    pca_rv = [0, 0.9, 0.99, 0.999]
    # for each retained variance, compute correlation
    for cur_pca_rv in pca_rv:
        # apply pca
        cur_ir = apply_pca(ir, n_components=cur_pca_rv)
        ax = cur_ir.T.plot()
        title = 'Original' if cur_pca_rv == 0 else 'Retained variance: '+str(cur_pca_rv)
        plt.title(title)
        plt.ylim([0, 0.05])
        plt.tight_layout()
        # save to file
        figure = ax.get_figure()
        figure.savefig(os.path.join(out_dir, '{:.3f}_ir.png'.format(cur_pca_rv)), dpi=dpi)
        # create gif 0 - 0.9
        if not cur_pca_rv == 0:
            cmd = 'convert -delay 100 -loop 0 {} {} {}'.format(
                os.path.join(out_dir, '0.000_ir.png'),
                os.path.join(out_dir, '{:.3f}_ir.png'.format(cur_pca_rv)),
                os.path.join(out_dir, 'comparison_with_{:.3f}.gif'.format(cur_pca_rv))
            )
            os.system(cmd)

    # # split legend on 2 columns
    # h,l = ax.get_legend_handles_labels()
    # ax.legend_.remove()
    # plt.gcf().legend(h,l, ncol=2)
