import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns  # ; sns.set_theme()
from data_preparation import Data


def corr_mat(df, tgt_idx, src_idx):
    # get variables
    src_df = df[src_idx]
    tgt_df = df[tgt_idx]
    # get corresponding feature matrices
    src = src_df.to_numpy()
    tgt = tgt_df.to_numpy()
    # get number of features
    nsrc = src.shape[1]
    ntgt = tgt.shape[1]
    # calculate Pearson's correlation
    pear = np.zeros((ntgt, nsrc))
    for i in range(ntgt):
        for j in range(nsrc):
            corr, _ = pearsonr(tgt[:, i], src[:, j])
            pear[i, j] = corr
    # convert to dataframe
    corr = pd.DataFrame(data=pear, index=tgt_idx, columns=src_idx)
    # return
    return corr


def corr_mat_image_top_n(corr_mat, n, sort=True, figsize=(5, 20), annotate=False):
    # increase font of seaborn
    sns.set(font_scale=3)
    # get best n
    cmat = corr_mat.to_numpy()
    best_idx = np.argsort(np.abs(cmat), axis=1)
    best_idx = best_idx[:, ::-1]  # invert for descending order
    if n > 0:
        best_idx = best_idx[:, :n]    # keep only n elements
    best_vals = np.take_along_axis(cmat, best_idx, axis=1)
    # sort rows from highest to lowest on highest correlation for each var
    if sort:
        order_idx = np.argsort(np.abs(best_vals[:, 0]))[::-1]
        best_idx = best_idx[order_idx, :]
        best_vals = best_vals[order_idx, :]
    # define labels
    best_var_names = np.tile(np.array(corr_mat.columns), (best_idx.shape[0], 1))
    best_var_names = np.take_along_axis(best_var_names, best_idx, axis=1)
    # convert to dataframe
    cols = ['Max'] if n == 1 else [str(i) for i in range(n)]
    res = pd.DataFrame(data=best_vals, index=corr_mat.index, columns=cols)
    # plot
    plt.figure(figsize=figsize)
    plt.clf()
    # write var names inside cells
    if annotate:
        ax = sns.heatmap(res, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, yticklabels=True)
        for t, cur_name in zip(ax.texts, best_var_names.ravel()):
            t.set_text(t.get_text() + ' (' + str(cur_name) + ')')
    else:
        ax = sns.heatmap(res, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, yticklabels=True)
    plt.tight_layout()
    figure = ax.get_figure()
    # return
    return figure


if __name__ == '__main__':
    # define settings
    dpi = 200
    out_dir = 'correlation'
    os.makedirs(out_dir, exist_ok=True)

    # get data
    data = Data()
    df, src_idx, tgt_idx = data.joined()

    # get correlation matrix
    corr = corr_mat(df, tgt_idx, src_idx)
    corr = corr.round(4)
    corr.to_csv(os.path.join(out_dir, 'corr_mat.csv'), sep=';')

    # save top 10
    top10 = corr_mat_image_top_n(corr, n=10, sort=True, figsize=(30, 20), annotate=False)
    top10.savefig(os.path.join(out_dir, 'top10.png'), dpi=dpi)
    # save top 1 annotated
    top1_ann = corr_mat_image_top_n(corr, n=1, sort=True, figsize=(20, 20), annotate=True)
    top1_ann.savefig(os.path.join(out_dir, 'top1_ann.png'), dpi=dpi)
    # save top 1 not annotated
    top1_na = corr_mat_image_top_n(corr, n=1, sort=True, figsize=(10, 20), annotate=False)
    top1_na.savefig(os.path.join(out_dir, 'top1_na.png'), dpi=dpi)
