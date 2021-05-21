import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from DatasetLucas import DatasetLucas
import seaborn as sns
import pandas as pd
from tqdm import tqdm
sns.set_style("dark")
from networks import Net


class GradCAM(object):
    def __init__(self, net, layers):
        self.net = net
        # set layers
        self.layers = layers
        # register forward hook
        for cur_layer in self.layers:
            cur_layer.register_forward_hook(self.fwd_hook)
        # register backward hook
        for cur_layer in self.layers:
            cur_layer.register_backward_hook(self.bwd_hook)
            # cur_layer.register_full_backward_hook(self.bwd_hook)
        # create placeholders
        self.activations = []
        self.gradients = []


    def fwd_hook(self, layer, inp, out):
        self.activations.append(out)


    def bwd_hook(self, layer, grad_input, grad_output):
        self.gradients.insert(0,grad_output[0])


    def gradcam_of_layer(self, cur_activations, cur_gradients, insz):
        # pool gradients
        pooled_gradients = torch.mean(cur_gradients, dim=[2])
        # weight activations
        cur_activations = cur_activations*pooled_gradients.unsqueeze(-1)
        # average the channels of the activations
        cur_heatmap = torch.mean(cur_activations, dim=1) #.unsqueeze(1)
        # relu on top of the cur_heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        cur_heatmap = torch.relu(cur_heatmap).detach()
        # normalize the cur_heatmap
        # mn = cur_heatmap.min(1).values.unsqueeze(1)
        mx = cur_heatmap.max(1).values.unsqueeze(1)
        cur_heatmap = cur_heatmap / mx
        # cur_heatmap = (cur_heatmap - mn) / (mx - mn)
        # cur_heatmap = (cur_heatmap - cur_heatmap.min()) / (cur_heatmap.max() - cur_heatmap.min())
        # interpolate
        cur_heatmap_t = cur_heatmap.unsqueeze(1).unsqueeze(1)
        cur_heatmap_t = F.interpolate(cur_heatmap_t, size=(1,insz), mode='bilinear', align_corners=True)
        cur_heatmap = cur_heatmap_t.squeeze(1).squeeze(1)
        # return
        return cur_heatmap



    def __call__(self, inp):
        # get net output to init heatmaps
        out = self.net(inp)
        # init accumulator
        heatmaps = torch.zeros(inp.size(0), out.size(-1), len(self.layers), inp.size(-1)) # B x V x L x S
        # for each output, backprop
        for i in range(out.shape[1]):
            # reset gradients
            self.activations = []
            self.gradients = []
            self.net.zero_grad()
            # forward
            out = self.net(inp)
            # backprop
            backstart = torch.zeros_like(out)
            backstart[:,i] = 1
            out.backward(backstart) #, retain_graph=True)
            # out[:,i].backward(torch.ones_like(out[:,i]), retain_graph=True)
            hms = None
            for cur_activations, cur_gradients in zip(self.activations, self.gradients):
                cur_activations = cur_activations.detach()
                cur_gradients = cur_gradients.detach()
                cur_hm = self.gradcam_of_layer(cur_activations, cur_gradients, inp.size(-1))
                cur_hm = cur_hm.unsqueeze(1)
                hms = cur_hm if hms is None else torch.cat((hms, cur_hm), 1)
            # append
            heatmaps[:,i] = hms
        # return heatmaps
        return heatmaps



if __name__ == '__main__':
    # import stuff
    from Experiment import Experiment
    # parse arguments
    parser = argparse.ArgumentParser()
    # checkpoint path
    parser.add_argument("-ckp", "--checkpoint", help="Checkpoint.",
    					default='', type=str)
    parser.add_argument("-out", "--out", help="Output filename.",
    					default='', type=str)
    # parse args
    args = parser.parse_args()
    # load model
    model = Experiment.load_from_checkpoint(args.checkpoint)
    model.eval()
    # define colormap
    cmap = matplotlib.cm.get_cmap('YlOrRd')
    # get dataloader
    dl = model.test_dataloader(return_coords=False)
    # get vars
    tgt_vars = model.conf['tgt_vars']
    # if out not specified, save in folder
    if args.out == '':
        args.out = os.path.join(os.path.dirname(args.checkpoint), 'test.csv')
    # get layers
    layers_names, layers_obj = zip(*[cur_name for cur_name in model.net.named_modules() if '.' in cur_name[0]])
    # get layers types as strings
    layers_types = [str(type(cur_layer)).split("'")[1].split('.')[-1] for cur_layer in layers_obj]
    # create gradcam object
    gc = GradCAM(model.net, layers_obj)
    # plt.figure(figsize=(10,100))
    # for each batch
    for src, tgt, bins, reg in dl:
        # compute prediction
        grads = gc(src)
        # for each sample
        for ns in range(grads.shape[0]):
            # clear plot
            plt.figure(figsize=(20,80))
            # plt.clf()
            # for each var
            for nv in range(len(tgt_vars)):
                # for each layer
                for nl in range(len(layers_names)):
                    # create subplot
                    plt.subplot(len(layers_names),len(tgt_vars),(nl+1)*len(tgt_vars))
                    # scatter plot signal with colors of gradcam
                    plt.scatter(dl.src_x, src[ns].squeeze(), c=cmap(grads[ns,nv,nl]))
                    # add title with current layer
                    plt.title(layers_names[nl] + ' ' + layers_types[nl])
            # plt.show()
            plt.suptitle(tgt_vars[nv])
            plt.tight_layout()
            plt.savefig('./boh/{:d}.png'.format(ns), dpi=100)
            if ns > 10: break
