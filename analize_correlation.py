import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns  # ; sns.set_theme()

# read file
df = pd.read_csv("data/tot.csv", sep=';')
df.set_index('sample')

# drop textual cols
df = df.drop(['campo', 'prof cm', 'res_Text'], axis=1)  # 'TIPO_CAMPIONE',

# get indexes of source vars and target vars
src_idx = df.columns[df.columns.str.contains('ir_', case=False)].to_list() + \
    df.columns[df.columns.str.contains('gamma_', case=False)].to_list()  # 2129

tgt_idx = df.columns[df.columns.str.contains('res_', case=False)].to_list()  # 15

# get variables
src_df = df[src_idx]
tgt_df = df[tgt_idx]

# get corresponding feature matrices
src = src_df.to_numpy()
tgt = tgt_df.to_numpy()

# print shapes
print(src.shape)
print(tgt.shape)

# get number of features
nsrc = src.shape[1]
ntgt = tgt.shape[1]

# calculate Pearson's correlation
pear = np.zeros((nsrc, ntgt))
for i in range(nsrc):
    for j in range(ntgt):
        corr, _ = pearsonr(src[:, i], tgt[:, j])
        pear[i, j] = corr

# convert
corr = pd.DataFrame(data=pear, index=src_idx, columns=tgt_idx)
corr = corr.round(2)
corr.to_csv('corr_mat.csv', sep=';')


# save correlation matrix
plt.clf()
plt.figure(figsize=(10, 20))
ax = sns.heatmap(corr, annot=False, cmap='coolwarm', cbar=True, xticklabels=True)
figure = ax.get_figure()
figure.savefig('corr_matrix.png', dpi=400)


# save only maximum correlations for each target variable
corrt = corr.T
res = corrt.max(axis=1)
# convert into dataframe
res = pd.DataFrame(data=res[:, np.newaxis], index=corrt.index, columns=['Max'])


plt.figure(figsize=(5, 20))
plt.clf()
ax = sns.heatmap(res, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, yticklabels=True)
plt.tight_layout()
figure = ax.get_figure()
figure.savefig('corr_matrix_compact.png', dpi=400)


# save only maximum correlations for each target variable in descent order
sns.set(font_scale=2)

plt.figure(figsize=(8, 20))
res_desc = res.reindex(res.Max.abs().sort_values(ascending=False).index)
plt.clf()
ax = sns.heatmap(res_desc, annot=True, fmt='.2f', cmap='coolwarm', cbar=False, yticklabels=True)
plt.tight_layout()
figure = ax.get_figure()
figure.savefig('corr_matrix_compact_desc.png', dpi=400)
