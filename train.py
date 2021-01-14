import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from DatasetLucas import DatasetLucas

# define test procedure
def test(net, ds):
    net.eval()
    loss = None
    for src,tgt in ds:
        # move in device
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        # compute prediction
        out = net(src)
        # compare
        cur_l = F.l1_loss(out,tgt).cpu()
        loss = cur_l if loss is None else torch.cat((loss,cur_l), axis=0)
    # compute average
    loss = loss.mean()
    net.train()
    return loss


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
    parser.add_argument("-ne", "--nepochs", help="Number of epochs.",
    					default=500, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate.",
    					default=1e-3, type=float)
    parser.add_argument("-dev", "--device", help="Device.",
    					default='cpu', type=str)
    parser.add_argument("-tcsv", "--train_csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_train.csv')
    parser.add_argument("-vcsv", "--val_csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_val.csv')
    args = parser.parse_args()
    # define network
    net = Net()
    net.to(args.device)
    # init loss
    # loss = nn.MSELoss()
    loss = F.l1_loss
    # init optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    # define datasets
    train_dataset = DatasetLucas(csv = args.train_csv, batch_size = args.batchsize)
    val_dataset = DatasetLucas(csv = args.val_csv, batch_size = args.batchsize)
    # init metric
    best_val = None
    # for each epoch
    for ne in range(args.nepochs):
        # for each element
        for i, (src, tgt) in enumerate(train_dataset):
            # move in device
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            # reset grads
            optimizer.zero_grad()
            # apply network
            out = net(src)
            # apply loss
            cur_l = loss(out, tgt)
            # print
            print('Loss = {:.4f} - Val: {:.4f}'.format(cur_l, best_val if best_val is not None else 0.0))
            # backward propagation
            cur_l.backward()
            # update weights
            optimizer.step()
        torch.save({'net':net.state_dict(),'opt':optimizer.state_dict()}, 'latest.pth')
        # test on pignoletto dataset
        dist = test(net, val_dataset)
        if best_val is None or dist < best_val:
            torch.save({'net':net.state_dict(),'opt':optimizer.state_dict()}, 'best.pth')
            best_val = dist
        # print('Distance with validation set: {:.4f}'.format(dist))
