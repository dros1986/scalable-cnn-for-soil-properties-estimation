import pandas as pd
import numpy as np
import argparse
import os
import pickle
# import torch
# import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import r2_score

# Miscellaneous
from sklearn import utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import PredefinedSplit



def get_src(df, src_prefix, fmin=None, fmax=None):
    ''' gets infrared signals using cols starting with src_prefix '''
    # get x points
    src_cols = [col for col in df.columns if col.startswith(src_prefix)]
    x = np.array([float(col[len(src_prefix):]) for col in src_cols])
    # sort x points in increasing order
    pos = np.argsort(x)
    src_cols = [src_cols[cur_pos] for cur_pos in pos]
    # extract x and y values
    src_x = x[pos]
    src_y = df[src_cols].to_numpy()
    # keep only frequences in range
    cond = np.ones(src_x.shape).astype(bool)
    if fmin is not None:
        # import ipdb; ipdb.set_trace()
        cond *= src_x>=fmin
    if fmax is not None:
        cond *= src_x<=fmax
    src_y = src_y[:,cond]
    src_cols = [cur_col for (cur_col, cur_cond) in zip(src_cols, cond) if cur_cond]
    src_x = src_x[cond]
    # convert to tensor
    # src_x = torch.from_numpy(src_x).float()
    # src_y = torch.from_numpy(src_y).float()
    src_x = src_x.astype(np.float32)
    src_y = src_y.astype(np.float32)
    # return
    return src_cols, src_x, src_y


def measure(y_pred, y_test):
    mae = np.mean(abs(y_test - y_pred))
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2



def train_model(model, x_train, y_train):
    # get number of features
    nfeats = 1 if y_train.ndim == 1 else y_train.shape[1]
    # train random forest
    if model == 'rf':
        print("Random Forest...")
        rf = RandomForestRegressor(max_features = nfeats, n_estimators = 200)
        rf.fit(x_train, y_train)
        return rf
    # train svr
    if model == 'svr':
        print("SVR...")
        svr = svm.SVR(kernel = "rbf", C = 1000, gamma = 0.01)
        svr.fit(x_train, y_train)
        return svr
    # train brt
    if model == 'brt':
        print("BRT...")
        brt = GradientBoostingRegressor(learning_rate = 0.1, min_samples_split = 6, n_estimators = 200)
        brt.fit(x_train, y_train)
        return brt



################################################################################

if __name__ == '__main__':
    # parse arguments
    # get arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-indir", "--indir", help="Input dir.",
    					default='/home/flavio/datasets/LucasLibrary/shared/', type=str)
    parser.add_argument("-outdir", "--outdir", help="Output dir.",
    					default='./sotaok', type=str)
    parser.add_argument("-train", "--train", help="If set, trains models.",
    					action='store_true')
    parser.add_argument("-render", "--render", help="If set, renders points in map.",
    					action='store_true')
    parser.add_argument("-map", "--map", help="Map.",
    					default='/home/flavio/Scaricati/ref-nuts-2021-01m.shp/NUTS_RG_01M_2021_4326.shp/NUTS_RG_01M_2021_4326.shp',
                        type=str)
    parser.add_argument("-tgt_vars", "--tgt_vars", help="Variables.",
    					default = ['coarse','clay','silt','sand','pH.in.CaCl2','pH.in.H2O','OC','CaCO3','N','P','K','CEC'],
                        nargs='+')
    parser.add_argument("-models", "--models", help="Models.",
    					default = ['rf','svr','brt'],
                        nargs='+')
    parser.add_argument("-fmin", "--fmin", help="Lowest band.",
                        default=450, type=float)
    parser.add_argument("-fmax", "--fmax", help="Highest band.",
                        default=None, type=float)
    # parse args
    args = parser.parse_args()

    # create output dir
    os.makedirs(args.outdir, exist_ok=True)

    if args.train:
        # load training data
        print("Reading train data...")
        lucas_train = pd.read_csv(os.path.join(args.indir, "lucas_dataset_train.csv"), sep=',')
        # for each variable
        for cur_tgt_var in args.tgt_vars:
            print('*************** {} ***************'.format(cur_tgt_var))
            # get training data
            #x_train = np.array(lucas_train.iloc[:,1:4201])
            cols, bands, x_train = get_src(lucas_train, 'spc.', fmin=args.fmin, fmax=args.fmax)
            y_train = np.array(lucas_train[cur_tgt_var])
            x_train = x_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            # train
            for model in args.models:
                # define output path
                cur_fn = os.path.join(args.outdir, '{}_{}.pkl'.format(cur_tgt_var, model))
                # if exists, skip training
                if os.path.isfile(cur_fn): continue
                # train
                cur_model = train_model(model, x_train, y_train)
                # save model
                pickle.dump(cur_model, open(cur_fn, 'wb'))
    else:
        # load test data
        print("Reading test data...")
        lucas_test = pd.read_csv(os.path.join(args.indir, "lucas_dataset_test.csv"))
        # define output
        df = pd.DataFrame(columns=['method', 'var', 'mae', 'mse', 'rmse', 'r2'])
        # for each variable
        for cur_tgt_var in args.tgt_vars:
            print('*************** {} ***************'.format(cur_tgt_var))
            # get data
            # x_test = np.array(lucas_test.iloc[:,1:4201])
            cols, bands, x_test = get_src(lucas_test, 'spc.', fmin=args.fmin, fmax=args.fmax)
            y_test = np.array(lucas_test[cur_tgt_var])
            x_test = x_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
            coords = lucas_test[['GPS_LAT','GPS_LONG']]
            # for each model
            for cur_model in args.models:
                # load model
                cur_fn = os.path.join(args.outdir, '{}_{}.pkl'.format(cur_tgt_var, cur_model))
                # model = pickle.load(open('K_rf.pkl','rb'))
                cur_m = pickle.load(open(cur_fn, 'rb'))
                # test
                pred = cur_m.predict(x_test)
                mae, mse, rmse, r2 = measure(pred, y_test)
                # append results
                df = df.append({'method': cur_model, 'var' : cur_tgt_var, 'mae': mae, 'mse': mse, 'rmse': rmse, 'r2':r2}, ignore_index=True)
                # save predictions
                cur_fn = os.path.join(args.outdir, '{}_{}.csv'.format(cur_tgt_var, cur_model))
                pred = pd.DataFrame(pred, columns=[cur_tgt_var])
                pred.to_csv(cur_fn, index=False)
                # render
                if args.render:
                    from Renderer import Renderer
                    # create renderer object
                    renderer = Renderer(args.map, crs=4326)
                    # concatenate coordinates
                    pred = pd.concat([coords, pred], axis=1)
                    # render
                    cur_outdir = os.path.join(args.outdir, cur_model)
                    os.makedirs(cur_outdir, exist_ok = True)
                    renderer.render(pred, [cur_tgt_var], out_dir=cur_outdir, \
                            spatial_res=(0.05, 0.05), lon_key='GPS_LONG', lat_key='GPS_LAT')


        # for each metric
        for cur_metric in ['mae', 'mse', 'rmse', 'r2']:
            # create output var
            df_out = None
            # group by method
            for name, group in df.groupby('method'): # 'var', 'mae', 'mse' ..
                cur_group = group[['var', cur_metric]]
                cur_group = cur_group.set_index('var')
                cur_group = cur_group.transpose()
                cur_group.insert(loc=0, column='method', value=name)
                cur_group = cur_group.round(4)
                if df_out is None:
                    df_out = cur_group
                else:
                    df_out = pd.concat([df_out, cur_group], ignore_index=False)
            # save matric-specific df
            df_out.to_csv(os.path.join(args.outdir,"sota_{}.csv".format(cur_metric)), index=False)
            df_out.to_latex(os.path.join(args.outdir,"sota_{}.tex".format(cur_metric)), float_format='%.4f', decimal='.', index=False)
