import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


class Quantizer(nn.Module):
    def __init__(self, nbins, nvars):
        super(Quantizer, self).__init__()
        # create parameters holders
        self.register_buffer('nbins', torch.tensor([nbins]))
        self.register_buffer('fmin', torch.zeros(nvars))
        self.register_buffer('fmax', torch.zeros(nvars))
        self.register_buffer('setup', torch.tensor([True]))


    def load_state_dict(self, state):
        self.nbins = state['nbins']
        self.fmin = state['fmin']
        self.fmax = state['fmax']
        self.setup[0] = False


    def forward(self, feats):
        # get number of vars
        nvars = feats.shape[1]
        # compute min and max of each variable, if not specified in init
        if self.setup[0]:
            self.fmin = feats.min(0)[0]
            self.fmax = feats.max(0)[0]
            self.setup[0] = False
        # create linears
        # linears = [torch.linspace(self.fmin[i], self.fmax[i], self.nbins) for i in range(nvars)]
        # define output variables
        bins = torch.zeros_like(feats).long()
        regs = torch.zeros_like(feats)
        # for each var
        for nv in range(nvars):
            # get current variable and related
            cur_min = self.fmin[nv]
            cur_max = self.fmax[nv]
            # cur_lin = linears[nv]
            cur_lin = torch.linspace(cur_min, cur_max, self.nbins.item())
            cur_var = feats[:,nv]
            cur_lin_step = cur_lin[1]-cur_lin[0]
            # expand current var and linear to match size
            cur_var_exp = cur_var.unsqueeze(1).repeat(1,self.nbins)
            cur_lin_exp = cur_lin.unsqueeze(0).repeat(cur_var.shape[0],1)
            # compute difference
            cur_diff = cur_lin_exp - cur_var_exp
            # round to 6th decimal to avoid near-zero elements. Otherwise gets wrong bin.
            cur_diff = np.around(cur_diff, decimals=6)
            # set negative difference to a high value so it won't be selected as mindist
            cur_diff[cur_diff>0] = cur_diff.min()
            # get bin of each sample
            cur_bin = cur_diff.max(1)[1]
            # get missing part as delta (regression part)
            cur_reg = cur_var - cur_lin_exp[torch.arange(cur_bin.shape[0]), cur_bin]
            cur_reg = torch.clamp(cur_reg, min=0)
            # normalize by bin size
            cur_reg = cur_reg / cur_lin_step
            # set output
            bins[:,nv] = cur_bin
            regs[:,nv] = cur_reg
        # return outputs
        return bins, regs


    def invert(self, bins, regs):
        # convert to long
        bins = bins.long()
        # convert to device
        self.fmin = self.fmin.to(bins.device)
        self.fmax = self.fmax.to(bins.device)
        # get number of variables
        nsamples = bins.shape[0]
        nvars = self.fmin.shape[0]
        # create output var
        feats = torch.zeros(nsamples, nvars).to(bins.device) #.as_type(regs.dtype)
        # for each var
        for nv in range(nvars):
            # get current variable and related
            cur_min = self.fmin[nv]
            cur_max = self.fmax[nv]
            cur_bin = bins[:,nv]
            cur_reg = regs[:,nv]
            # create corresponding linear space and get linear step
            cur_lin = torch.linspace(cur_min, cur_max, self.nbins.item()).to(bins.device)
            cur_lin_step = cur_lin[1]-cur_lin[0]
            # expand current linear to match size
            cur_lin_exp = cur_lin.unsqueeze(0).repeat(nsamples,1)
            # select bin value according to cur_bins
            cur_var = cur_lin_exp[torch.arange(nsamples), cur_bin]
            # add regression part
            cur_var = cur_var + cur_reg*cur_lin_step
            # append to feats
            feats[:,nv] = cur_var
        # return
        return feats


if __name__ == '__main__':
    # create quantizer
    quant = Quantizer(nbins=10)
    # read csv
    df = pd.read_table('./data/LUCAS.SOIL_corr_FULL_val.csv', sep=',')
    # keep only few wars for testing
    df = df[['pH.in.CaCl2','pH.in.H2O','OC','CaCO3']]
    # obtain features
    feats = torch.from_numpy(df.to_numpy()).float()
    # quantize
    bins, regs = quant.quantize(feats)
    # unquantize
    feats_rec = quant.unquantize(bins, regs)
    # check everything is ok
    assert((feats-feats_rec).mean().item() < 1e-3)
