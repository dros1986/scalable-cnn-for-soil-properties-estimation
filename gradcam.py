import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from DatasetLucas import DatasetLucas
import seaborn as sns
import pandas as pd
from tqdm import tqdm
sns.set_style("dark")
from train import Net


class GradCAM(object):
    def __init__(self, net, layer):
        self.net = net
        # get layer
        self.layer = layer
        # register forward hook
        self.layer.register_forward_hook(self.fwd_hook)
        # register backward hook
        self.layer.register_backward_hook(self.bwd_hook)
        # create placeholders
        self.activations = None
        self.gradients = None

    def fwd_hook(self, layer, inp, out):
        self.activations = out

    def bwd_hook(self, layer, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, inp):
        self.activations = None
        out = self.net(inp)
        # init accumulator
        heatmaps = []
        # for each output, backprop
        for i in range(out.shape[1]):
            # reset gradients
            self.gradients = None
            self.net.zero_grad()
            # forward
            out = self.net(inp)
            # backprop
            # out[:,i].mean().backward(retain_graph=True)
            out[:,i].backward(torch.ones_like(out[:,i]), retain_graph=True)
            # pool gradients
            pooled_gradients = torch.mean(self.gradients, dim=[2, 3])
            # weight activations
            cur_activations = self.activations*pooled_gradients.unsqueeze(-1).unsqueeze(-1)
            # average the channels of the activations
            # cur_heatmap = torch.mean(cur_activations, dim=1).squeeze()
            cur_heatmap = torch.mean(cur_activations, dim=1).unsqueeze(1)
            # relu on top of the cur_heatmap
            # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
            cur_heatmap = torch.relu(cur_heatmap).detach()
            # normalize the cur_heatmap
            cur_heatmap = (cur_heatmap - cur_heatmap.min(3).values.unsqueeze(3)) / \
                        (cur_heatmap.max(3).values.unsqueeze(3) - cur_heatmap.min(3).values.unsqueeze(3))
            # interpolate to match size of input signal
            cur_heatmap = F.interpolate(cur_heatmap, size=(inp.size(2),inp.size(3)), mode='bilinear', align_corners=True)
            # remove unuseful dimensions
            cur_heatmap = cur_heatmap.squeeze(1).squeeze(1).detach()
            # append
            heatmaps.append(cur_heatmap)
        # return values
        return heatmaps




if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-ne", "--number_of_elements", help="number of elements to save.",
    					default=10, type=int)
    parser.add_argument("-dev", "--device", help="Device.",
    					default='cpu', type=str)
    parser.add_argument("-csv", "--csv", help="Lucas test csv file.",
    					default='./data/LUCAS.SOIL_corr_FULL_val.csv')
    parser.add_argument('-last', '--latest', action='store_true')
    parser.add_argument("-exp", "--experiment", help="Name of experiment.",
    					default='experiment1', type=str)
    args = parser.parse_args()
    # define network
    net = Net()
    net.to(args.device)
    net.eval()
    # load weights and vars
    wfn = os.path.join(args.experiment, 'latest.pth' if args.latest else 'best.pth')
    state = torch.load(wfn)
    net.load_state_dict(state['net'])
    vars = state['vars']
    # define dataset
    dataset = DatasetLucas(csv = args.csv, batch_size = args.number_of_elements, drop_last=False, vars=vars)
    # get x
    x = dataset.src_x.numpy()
    tgt_names = dataset.tgt_names
    # init gradcam
    gradcam = GradCAM(net, net.b6)
    # set colormap
    cmap = matplotlib.cm.get_cmap('YlOrRd') # Blues YlOrRd jet
    # for each element
    for i, (src, tgt) in enumerate(dataset):
        # move in device
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        # get heatmaps
        heatmaps = gradcam(src)
        # for each target variable
        for nv in tqdm(range(tgt.size(1))):
            # get current variable name
            cur_var_name = tgt_names[nv]
            # for each sample
            for i in range(src.size(0)):
                # get current gradcams
                cur_sample = src[i].squeeze(0).squeeze(0).cpu().numpy()
                cur_heatmap = heatmaps[nv][i].cpu().numpy()
                # create dataframe
                df = pd.DataFrame({'wavelength': x, 'signal': cur_sample, 'heatmap':cur_heatmap})
                # define colors
                plt.figure(0)
                plt.clf()
                # cur_colors = np.vstack([np.array(cmap(col)) for col in cur_heatmap])
                # ax = plt.scatter(x, cur_sample, s=8, c=cur_colors)
                ax = sns.scatterplot(data=df, x='wavelength', y='signal', hue='heatmap', palette=cmap, linewidth=0) #, sizes=40)
                # ax = sns.lineplot(data=df, x='wavelength', y='signal', hue='heatmap', palette=cmap) #, sizes=40)
                plt.grid()
                plt.title(cur_var_name)
                plt.tight_layout()
                figure = ax.get_figure()
                out_fn = os.path.join(args.experiment, 'gradcam', cur_var_name, str(i) + '.png')
                os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                figure.savefig( out_fn, dpi=100)
                # plt.show()
        break
