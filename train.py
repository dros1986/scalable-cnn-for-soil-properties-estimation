import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from DatasetLucas import DatasetLucas
from DatasetPignoletto import DatasetPignoletto

# define network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.b1 = self.block(1,16)
        self.b2 = self.block(16,32)
        self.b3 = self.block(32,64)
        self.b4 = self.block(64,128)
        self.b5 = self.block(128,256)
        self.b6 = self.block(256,128)
        self.final = nn.Linear(3968,7)


    def block(self, ch_in, ch_out, sz=(1,3), st=(1,2), pad=0):
        return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=sz, stride=st, padding=pad),
    		nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = x.view(x.size(0),-1)
        x = self.final(x)
        return x

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--regen", help="Regenerate images using the model specified.",
    					default="")
    parser.add_argument("-ps", "--patchsize", help="Dimension of the patch.",
    					default=8, type=int)
    parser.add_argument("-bs", "--batchsize", help="Batch size.",
    					default=100, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate.",
    					default=1e-3, type=float)
    parser.add_argument("-dev", "--device", help="Device.",
    					default='cpu', type=str)
    parser.add_argument("-lcsv", "--lucas_csv", help="Lucas csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_CLEAN.csv')
    parser.add_argument("-tsl", "--test_list", help="Test list.",
    					default='./datasets/places-instagram/test-list.txt')
    args = parser.parse_args()
    # define network
    net = Net()
    net.to(args.device)
    # init loss
    loss = nn.MSELoss()
    # init optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    # define datasets
    lucas = DatasetLucas(csv = args.lucas_csv, batch_size = args.batchsize)
    pignoletto = DatasetPignoletto()
    # # get first element of both datasets
    # lu_ir, lu_tg  = iter(lucas).__next__()
    # pi_ir, pi_tg = iter(pignoletto).__next__()
    # lu_ir = lu_ir.squeeze(1).squeeze(1)
    # pi_ir = pi_ir.squeeze(1).squeeze(1)
    # # plot
    # plt.plot(lu_ir[0], label='lucas')
    # plt.plot(pi_ir[0], label='pignoletto')
    # plt.legend()
    # plt.show()
    # for each element
    for i, (src, tgt) in enumerate(lucas): # 2101 7
        # move in device
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        # reset grads
        optimizer.zero_grad()
        # apply network
        out = net(src)
        print(src.size())
        print(out.size())
        # apply loss
        cur_l = loss(out, tgt)
        # print
        print('Loss = {:.4f}'.format(cur_l))
        # backward propagation
        l.backward()
        # update weights
        optimizer.step()
