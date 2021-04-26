import os
import argparse
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regen import regen
from sklearn.metrics import r2_score



def var_index_to_dataframe(score, tgt_vars, glob=None, higher_is_better=False):
    # define output dataframe
    df = pd.DataFrame(columns=['variable', 'value'])
    # put variables inside dataframe
    for (vn,vl) in zip(tgt_vars, score.cpu().numpy().tolist()):
        df = df.append({'variable': vn, 'value': vl}, ignore_index=True)
    # sort by value
    df.sort_values('value', ascending=False if higher_is_better else True, inplace=True)
    # append average
    df = df.append({'variable': 'avg', 'value': score.mean().item()}, ignore_index=True)
    # append global index
    df = df.append({'variable': 'global', 'value': glob.item()}, ignore_index=True)
    # round to 4 decimals
    df = df.round(4)
    # set variable name as index
    df.set_index('variable', inplace=True)
    # return dataframe
    return df


def test_batch(out, tgt, tgt_vars):
    # compute difference
    pla_diff = out - tgt
    abs_diff = torch.abs(pla_diff)
    sqr_diff = torch.pow(pla_diff, 2)
    # compute mae, mse, rmse, r2
    tot_mae = abs_diff.mean()
    tot_mse = sqr_diff.mean()
    tot_rmse = torch.sqrt(tot_mse)
    tot_r2 = torch.tensor(r2_score(tgt.cpu(), out.cpu())).float()
    # get variable stats
    var_mae = abs_diff.mean(1)
    var_mse = sqr_diff.mean(1)
    var_rmse = torch.sqrt(var_mse)
    var_r2 = torch.tensor([r2_score(tgt[:,v].cpu(), out[:,v].cpu()) for v in range(tgt.size(1))])
    # create output
    ris = {
        'mae': var_index_to_dataframe(var_mae, tgt_vars, glob=tot_mae, higher_is_better=False),
        'mse': var_index_to_dataframe(var_mse, tgt_vars, glob=tot_mse, higher_is_better=False),
        'rmse': var_index_to_dataframe(var_rmse, tgt_vars, glob=tot_rmse, higher_is_better=False),
        'r2': var_index_to_dataframe(var_r2, tgt_vars, glob=tot_r2, higher_is_better=True),
    }
    # return
    return ris


def test(net, src_norm, tgt_norm, tgt_quant, test_dataset, tgt_vars, device='cuda'):
    # regenerate
    dfout, dftgt = regen(net, src_norm, tgt_norm, tgt_quant, test_dataset, tgt_vars, device='cuda')
    # check everything is consistent
    for cout, ctgt in zip(dfout.columns, dftgt.columns): assert(cout == ctgt)
    # convert to pytorch
    out = torch.from_numpy(dfout.to_numpy()).float()
    tgt = torch.from_numpy(dftgt.to_numpy()).float()
    # test
    return test_batch(out, tgt, tgt_vars)

    # # compute difference
    # pla_diff = torch.from_numpy(dfout.to_numpy() - dftgt.to_numpy()).float()
    # abs_diff = torch.abs(pla_diff)
    # sqr_diff = torch.pow(pla_diff, 2)
    # # compute mae, mse, rmse, r2
    # tot_mae = abs_diff.mean()
    # tot_mse = sqr_diff.mean()
    # tot_rmse = torch.sqrt(tot_mse)
    # tot_r2 = torch.tensor(r2_score(dftgt.to_numpy(), dfout.to_numpy())).float()
    # # get variable stats
    # var_mae = abs_diff.mean(1)
    # var_mse = sqr_diff.mean(1)
    # var_rmse = torch.sqrt(var_mse)
    # var_r2 = torch.tensor([r2_score(dftgt[v].to_numpy(), dfout[v].to_numpy()) for v in tgt_vars]).float()
    # # create output
    # ris = {
    #     'mae': var_index_to_dataframe(var_mae, tgt_vars, glob=tot_mae, higher_is_better=False),
    #     'mse': var_index_to_dataframe(var_mse, tgt_vars, glob=tot_mse, higher_is_better=False),
    #     'rmse': var_index_to_dataframe(var_rmse, tgt_vars, glob=tot_rmse, higher_is_better=False),
    #     'r2': var_index_to_dataframe(var_r2, tgt_vars, glob=tot_r2, higher_is_better=True),
    # }
    # # return
    # return ris

#
# def is_better(cur_ris_df, best_ris_df, metric_name):
#         # if best val is empty, return true
#         if best_ris_df is None: return True
#         # get current and best value
#         cur_val = cur_ris_df[metric_name].loc['global'].value
#         best_val = best_ris_df[metric_name].loc['global'].value
#         # compare
#         if metric_name.lower() == 'r2':
#             if cur_val > best_val: return True
#         else:
#             if cur_val < best_val: return True
#         return False
