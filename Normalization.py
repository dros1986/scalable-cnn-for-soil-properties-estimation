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
        # return standardized features
        return feats*vr + mu



class GlobalStandardization(nn.Module):
    ''' Standardizes using global mean and variance '''
    def __init__(self):
        super(GlobalStandardization, self).__init__()
        # init parameters
        self.register_buffer('mu', torch.tensor([]))
        self.register_buffer('vr', torch.tensor([]))

    def load_state_dict(self, state):
        self.mu = state['mu']
        self.vr = state['vr']

    def forward(self, feats):
        if self.mu.numel() == 0 or self.vr.numel() == 0:
            self.mu = feats.mean()
            self.vr = feats.var()
        # move to proper device
        if not (self.mu.device == feats.device or self.vr.device == feats.device):
            self.mu = self.mu.to(feats.device)
            self.vr = self.vr.to(feats.device)
        # return standardized features
        return (feats - self.mu) / self.vr

    def invert(self, feats):
        # move to proper device
        if not (self.mu.device == feats.device or self.vr.device == feats.device):
            self.mu = self.mu.to(feats.device)
            self.vr = self.vr.to(feats.device)
        # return features
        return feats*self.vr + self.mu



class VariableStandardization(nn.Module):
    ''' Standardizes each variable independently '''
    def __init__(self):
        super(VariableStandardization, self).__init__()
        # init parameters
        self.register_buffer('mu', torch.tensor([]))
        self.register_buffer('vr', torch.tensor([]))


    def load_state_dict(self, state):
        self.mu = state['mu']
        self.vr = state['vr']

    def forward(self, feats):
        if self.mu.numel() == 0 or self.vr.numel() == 0:
            self.mu = feats.mean(0).unsqueeze(0)
            self.vr = feats.var(0).unsqueeze(0)
        # move to proper device
        if not (self.mu.device == feats.device or self.vr.device == feats.device):
            self.mu = self.mu.to(feats.device)
            self.vr = self.vr.to(feats.device)
        # return standardized features
        return (feats - self.mu) / self.vr

    def invert(self, feats):
        # move to proper device
        if not (self.mu.device == feats.device or self.vr.device == feats.device):
            self.mu = self.mu.to(feats.device)
            self.vr = self.vr.to(feats.device)
        # return features
        return feats*self.vr + self.mu



if __name__ == '__main__':
    std = GlobalStandardization()
    # parameters defined in first call (on train)
    print(std.state_dict())
    std(torch.rand(50,3)).var()
    print(std.state_dict())
    # load_state_dict overrided to deal with size mismatch
    torch.save(std.state_dict(), 'prova.pth')
    state = torch.load('prova.pth')
    # import ipdb; ipdb.set_trace()
    newstd = GlobalStandardization()
    newstd.load_state_dict(state)
    print(newstd.state_dict())
