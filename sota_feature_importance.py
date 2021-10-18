import pandas as pd
import numpy as np
import argparse
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

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

# seaborn
import seaborn as sns
sns.set_style("dark")


def onsignal(signal, importance, fn, sz=4200):
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


# feature importance based on mean decrease in impurity
def rf_decrease_in_impurity(forest, sz=4200):
    # get feature importance
    importances = forest.feature_importances_
    std = np.std([
        tree.feature_importances_ for tree in forest.estimators_], axis=0)
    # define features name
    feature_names = [str(i) for i in range(sz)]
    # convert to pandas
    forest_importances = pd.Series(importances, index=feature_names)
    # return
    return forest_importances



def rf_feature_permutation(forest, x_test, y_test, sz=4200):
    feature_names = [str(i) for i in range(sz)]
    result = permutation_importance(forest, x_test, y_test, n_repeats=10, random_state=42, n_jobs=8)
    forest_importances = pd.Series(result.importances_mean, index=feature_names)
    return forest_importances



if __name__ == '__main__':
    # parse arguments
    # get arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-dsdir", "--dsdir", help="Dataset dir.",
    					default='/home/flavio/datasets/LucasLibrary/shared/', type=str)
    parser.add_argument("-ckdir", "--ckdir", help="Checkpoints dir.",
    					default='./sotaok', type=str)
    parser.add_argument("-outdir", "--outdir", help="Output dir.",
    					default='./sotaok', type=str)
    parser.add_argument("-tgt_vars", "--tgt_vars", help="Variables.",
    					default = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
                        nargs='+')
    parser.add_argument("-models", "--models", help="Models.",
                        default = ['rf'],
    					# default = ['rf','svr','brt'],
                        nargs='+')
    # parse args
    args = parser.parse_args()

    # create output dir
    os.makedirs(args.outdir, exist_ok=True)

    # load test data
    print("Reading test data...")
    lucas_test = pd.read_csv(os.path.join(args.dsdir, "lucas_dataset_test.csv"))

    # define output
    df = pd.DataFrame(columns=['method', 'var', 'mae', 'mse', 'rmse', 'r2'])
    # for each variable
    for cur_tgt_var in args.tgt_vars:
        print('*************** {} ***************'.format(cur_tgt_var))
        # get data
        x_test = np.array(lucas_test.iloc[:,1:4201])
        y_test = np.array(lucas_test[cur_tgt_var])
        coords = lucas_test[['GPS_LAT','GPS_LONG']]
        # for each model
        for cur_model in args.models:
            # load model
            cur_fn = os.path.join(args.outdir, '{}_{}.pkl'.format(cur_tgt_var, cur_model))
            cur_m = pickle.load(open(cur_fn, 'rb'))
            # get feature importance
            if isinstance(cur_m, RandomForestRegressor):
                importance = rf_decrease_in_impurity(cur_m)
                # importance = rf_feature_permutation(cur_m, x_test, y_test, sz=4200)
            elif isinstance(cur_m, svm.SVR):
                importance = cur_m.coef_[0]
            elif isinstance(cur_m, GradientBoostingRegressor):
                importance = cur_m.feature_importances_
            # draw it
            cur_out_dir = os.path.join(args.outdir, 'fi_{}'.format(cur_model))
            out_fn = os.path.join(cur_out_dir, cur_tgt_var + '.png')
            os.makedirs(cur_out_dir, exist_ok=True)
            onsignal(x_test[0], importance, out_fn, sz=4200)
