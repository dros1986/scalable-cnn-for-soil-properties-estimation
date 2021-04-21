import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class InstanceStandardization(nn.Module):
    ''' Standardizes using sample mean and variance '''
    def __init__(self):
        super(InstanceStandardization, self).__init__()

    def forward(self, feats):
        mu = feats.mean(1).unsqueeze(1)
        vr = feats.var(1).unsqueeze(1)
        return (feats - mu) / vr

    def invert(self, feats):
        mu = feats.mean(1).unsqueeze(1)
        vr = feats.var(1).unsqueeze(1)
        return feats*vr + mu



class GlobalStandardization(nn.Module):
    ''' Standardizes using global mean and variance '''
    def __init__(self, mu = None, vr = None):
        super(GlobalStandardization, self).__init__()
        # init parameters
        self.register_buffer('mu', torch.tensor([]))
        self.register_buffer('vr', torch.tensor([]))
        # set values
        self.mu = mu
        self.vr = vr

    def forward(self, feats):
        if self.mu == None or self.vr == None:
            self.mu = feats.mean()
            self.vr = feats.var()
        return (feats - self.mu) / self.vr

    def invert(self, feats):
        return feats*self.vr + self.mu



class VariableStandardization(nn.Module):
    ''' Standardizes each variable independently '''
    def __init__(self, mu = None, vr = None):
        super(VariableStandardization, self).__init__()
        # init parameters
        self.register_buffer('mu', torch.tensor([]))
        self.register_buffer('vr', torch.tensor([]))
        # set values
        self.mu = mu
        self.vr = vr

    def forward(self, feats):
        if self.mu == None or self.vr == None:
            self.mu = feats.mean(0).unsqueeze(0)
            self.vr = feats.var(0).unsqueeze(0)
        return (feats - self.mu) / self.vr

    def invert(self, feats):
        return feats*self.vr + self.mu



if __name__ == '__main__':
    std = GlobalStandardization()
    print(std.state_dict())
    std(torch.rand(50,3)).var()
    print(std.state_dict())
