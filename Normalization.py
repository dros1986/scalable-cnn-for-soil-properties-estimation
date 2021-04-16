import os
import torch
import pandas as pd
import numpy as np


class InstanceStandardization(object):
    ''' Standardizes using sample mean and variance '''
    def normalize(self, feats):
        mu = feats.mean(1).unsqueeze(1)
        vr = feats.var(1).unsqueeze(1)
        return (feats - mu) / vr

    def get_state(self): return {}
    def set_state(self, state): return


class GlobalStandardization(object):
    ''' Standardizes using global mean and variance '''
    def __init__(self, mu = None, vr = None):
        self.mu = mu
        self.vr = vr

    def normalize(self, feats):
        if self.mu == None or self.vr == None:
            self.mu = feats.mean()
            self.vr = feats.var()
        return (feats - self.mu) / self.vr

    def get_state(self):
        return {'mu':self.mu, 'vr':self.vr}

    def set_state(self, state):
        self.mu = state['mu']
        self.vr = state['vr']


class VariableStandardization(object):
    ''' Standardizes each variable independently '''
    def __init__(self, mu = None, vr = None):
        self.mu = mu
        self.vr = vr

    def normalize(self, feats):
        if self.mu == None or self.vr == None:
            self.mu = feats.mean(0).unsqueeze(0)
            self.vr = feats.var(0).unsqueeze(0)
        return (feats - self.mu) / self.vr

    def get_state(self):
        return {'mu':self.mu, 'vr':self.vr}

    def set_state(self, state):
        self.mu = state['mu']
        self.vr = state['vr']
