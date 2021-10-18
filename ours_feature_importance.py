import pandas as pd
import numpy as np
import argparse
import os
import torch
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# Metrics
from sklearn.metrics import r2_score

# Miscellaneous
from sklearn import utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import PredefinedSplit

# our stuff
from Experiment import Experiment
from GradCAM import GradCAM


def onsignal(signal, importance, fn): #, sz=4200):
    sz = signal.shape[0]
    # define colormap
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    # normalize between 0 and 1
    importance = (importance - importance.min()) / (importance.max() - importance.min())
    # plot heatmap
    plt.figure(figsize=(10,5))
    plt.scatter(np.array(range(sz)), signal, c=cmap(importance))
    plt.locator_params(axis='x', nbins=20)
    plt.tight_layout()
    # show
    plt.savefig(fn, dpi=100)



if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-ckp", "--checkpoint", help="Checkpoint.",
    					default='', type=str)
    parser.add_argument("-outdir", "--outdir", help="Output directory.",
    					default='', type=str)
    parser.add_argument("-layer", "--layer", help="Name of layer. down.[0-53] o emb.[0-3]",
    					default='down.53', type=str)
    # parse args
    args = parser.parse_args()
    # load model
    model = Experiment.load_from_checkpoint(args.checkpoint)
    # model.eval()
    # define colormap
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    # get dataloader
    dl = model.test_dataloader(return_coords=False)
    # get vars
    tgt_vars = model.conf['tgt_vars']
    # if out not specified, save in folder
    # if args.out == '':
    #     args.out = os.path.join(os.path.dirname(args.checkpoint), 'test.csv')
    # get layers
    layers_names, layers_obj = zip(*[cur_name for cur_name in model.net.named_modules() if '.' in cur_name[0]])
    # get layers types as strings
    layers_types = [str(type(cur_layer)).split("'")[1].split('.')[-1] for cur_layer in layers_obj]
    # select only some
    i = layers_names.index(args.layer)
    cur_layer_type = layers_types[i]
    print('Layer {} is of type {}'.format(args.layer, cur_layer_type))
    layers_names, layers_obj = layers_names[i:i+1], layers_obj[i:i+1]
    layers_types = layers_types[i:i+1]
    # create gradcam object
    gc = GradCAM(model.net, layers_obj)
    # all grads
    all_grads = None
    # first_signal
    first_signal = None
    # for each batch
    for src, tgt, bins, reg in tqdm(dl):
        # get first signal
        if first_signal is None:
            first_signal = src[0,0,0].cpu().numpy()
        # compute prediction
        grads = gc(src)
        # remove nans
        grads = grads[~torch.any(grads.view(grads.shape[0],-1).isnan(),dim=1)]
        grads = grads.detach().cpu()
        if all_grads is None:
            all_grads = grads
        else:
            all_grads = torch.cat((all_grads, grads),1)
    # mean over samples BxVxLxS
    all_grads = all_grads.mean(0)
    # remove layer dimension
    all_grads = all_grads[:,0]
    # set importance
    importance = all_grads.numpy()
    # plot for each variable
    for i in range(len(tgt_vars)):
        # get current variable name
        cur_tgt_var = tgt_vars[i]
        # get current importance
        cur_importance = importance[i]
        # draw it
        cur_out_dir = os.path.join(args.outdir, 'fi_ours_{}_{}'.format(args.layer, cur_layer_type))
        out_fn = os.path.join(cur_out_dir, cur_tgt_var + '.png')
        os.makedirs(cur_out_dir, exist_ok=True)
        # import ipdb; ipdb.set_trace()
        onsignal(first_signal, cur_importance, out_fn) #, sz=4200)
