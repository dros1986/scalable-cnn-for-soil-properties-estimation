import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
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
        with torch.no_grad():
            out = net(src)
        # compare
        # cur_l = F.l1_loss(out,tgt).cpu().unsqueeze(0)
        cur_l = F.l1_loss(out,tgt,reduction='none').cpu()
        loss = cur_l if loss is None else torch.cat((loss,cur_l), axis=0)
    # compute average
    var_loss = loss.mean(0)
    tot_loss = loss.mean()
    # fill DataFrame
    var_names = ds.tgt_names
    df = pd.DataFrame(columns=['variable', 'value'])
    for (vn,vl) in zip(var_names, var_loss):
        df = df.append({'variable': vn, 'value': vl.item()}, ignore_index=True)
    df = df.append({'variable': 'avg', 'value': tot_loss.item()}, ignore_index=True)
    df = df.round(4)
    df.set_index('variable', inplace=True)
    # net back to train
    net.train()
    return df
    # return var_loss, tot_loss


# # define network
# class Net(nn.Module):
#     def __init__(self, dropout=0):
#         super(Net, self).__init__()
#         self.b1 = self.block(1,16)
#         self.b2 = self.block(16,32)
#         self.b3 = self.block(32,64)
#         self.b4 = self.block(64,128)
#         self.b5 = self.block(128,128)
#         self.b6 = self.block(128,128)
#         self.b7 = self.block(128,128)
#         self.b8 = self.block(128,128)
#         self.b9 = self.block(128,128)
#         self.b10 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=(1,4), stride=1, padding=(0,0)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )
#         self.dropout = nn.Dropout(p=dropout)
#         self.final = nn.Sequential(
#                         nn.Linear(128, 128),
#                         nn.ReLU(inplace=True),
#                         nn.Linear(128, 12))
#
#
#     def block(self, ch_in, ch_out, sz=(1,3), st=(1,1), pad=(0,0)):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_in, kernel_size=(1,3), stride=1, padding=(0,0)),
#             nn.BatchNorm2d(ch_in),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(ch_in, ch_out, kernel_size=sz, stride=st, padding=pad),
#     		nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
#         )
#
#
#     def forward(self, x):
#         x = self.b1(x)
#         x = self.b2(x)
#         x = self.b3(x)
#         x = self.b4(x)
#         x = self.b5(x)
#         x = self.b6(x)
#         x = self.b7(x)
#         x = self.b8(x)
#         x = self.b9(x)
#         x = self.b10(x)
#         x = x.view(x.size(0),-1)
#         x = self.dropout(x)
#         x = self.final(x)
#         return x


# define network
class Net(nn.Module):
    def __init__(self, dropout=0):
        super(Net, self).__init__()
        self.b1 = self.block(1,8)
        self.b2 = self.block(8,16)
        self.b3 = self.block(16,16)
        self.b4 = self.block(16,32)
        self.b5 = self.block(32,32)
        self.b6 = self.block(32,64)
        self.b7 = self.block(64,64)
        self.b8 = self.block(64,64)
        self.b9 = self.block(64,64)
        self.b10 = self.block(64,64)
        self.b11 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1,2), stride=1, padding=(0,0)),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.final = nn.Sequential(
                        nn.Linear(64, 26), # 26
                        nn.ReLU(inplace=True),
                        nn.Linear(26, 12))


    def block(self, ch_in, ch_out, sz=(1,3), st=(1,1), pad=(0,0)):
        return nn.Sequential(
            # nn.Conv2d(ch_in, ch_in, kernel_size=(1,3), stride=1, padding=(0,0)),
            # nn.BatchNorm2d(ch_in),
            # nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=sz, stride=st, padding=pad),
    		nn.BatchNorm2d(ch_out, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
            nn.AvgPool2d(kernel_size=(1,2), stride=(1,2))
        )


    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = self.b9(x)
        x = self.b10(x)
        x = self.b11(x)
        x = x.view(x.size(0),-1)
        # x = self.dropout(x)
        x = self.final(x)
        return x


# # define network
# class Net(nn.Module):
#     def __init__(self, dropout=0):
#         super(Net, self).__init__()
#         self.b1 = self.block(1,16)
#         self.b2 = self.block(16,32)
#         self.b3 = self.block(32,64)
#         self.b4 = self.block(64,128)
#         self.b5 = self.block(128,128)
#         self.b6 = self.block(128,128)
#         self.b7 = self.block(128,128)
#         self.b8 = self.block(128,128)
#         self.b9 = self.block(128,128)
#         self.b10 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=(1,5), stride=1, padding=(0,0)),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#         )
#         self.dropout = nn.Dropout(p=dropout)
#         self.final = nn.Sequential(
#                         # nn.Linear(128, 128),
#                         # nn.ReLU(inplace=True),
#                         nn.Linear(128, 12))
#
#
#     def block(self, ch_in, ch_out, sz=(1,3), st=(1,1), pad=(0,0)):
#         return nn.Sequential(
#             nn.Conv2d(ch_in, ch_out, kernel_size=(1,5), stride=2, padding=(0,0)),
#             nn.BatchNorm2d(ch_out),
#             nn.ReLU(inplace=True)
#         )


    # def forward(self, x):
    #     x = self.b1(x)
    #     x = self.b2(x)
    #     x = self.b3(x)
    #     x = self.b4(x)
    #     x = self.b5(x)
    #     x = self.b6(x)
    #     x = self.b7(x)
    #     x = self.b8(x)
    #     x = self.b9(x)
    #     x = self.b10(x)
    #     x = x.view(x.size(0),-1)
    #     x = self.dropout(x)
    #     x = self.final(x)
    #     return x

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batchsize", help="Batch size.",
    					default=100, type=int)
    parser.add_argument("-ne", "--nepochs", help="Number of epochs.",
    					default=500, type=int)
    parser.add_argument("-lr", "--learning_rate", help="Learning rate.",
    					default=1e-3, type=float)
    parser.add_argument("-wd", "--weight_decay", help="Weight decay.",
    					default=0, type=float)
    parser.add_argument("-do", "--dropout", help="Dropout.",
    					default=0, type=float)
    parser.add_argument("-norm", "--norm_type", help="Normalization used in nir/swir.",
                        default='std_instance', type=str)
    parser.add_argument("-dev", "--device", help="Device.",
    					default='cpu', type=str)
    parser.add_argument("-exp", "--experiment", help="Name of experiment.",
    					default='experiment1', type=str)
    parser.add_argument("-tcsv", "--train_csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_train.csv')
    parser.add_argument("-vcsv", "--val_csv", help="Lucas train csv file.",
    					default='/home/flavio/datasets/LucasLibrary/LucasTopsoil/LUCAS.SOIL_corr_FULL_val.csv')
    args = parser.parse_args()
    # create output dir
    os.makedirs(args.experiment, exist_ok=True)
    # define network
    net = Net(dropout = args.dropout)
    net.to(args.device)
    # init loss
    # loss = nn.MSELoss()
    loss = F.l1_loss
    # init optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # define datasets
    train_dataset = DatasetLucas(csv = args.train_csv, batch_size = args.batchsize, norm_type = args.norm_type, \
                                drop_last=True)
    vars = train_dataset.get_vars()
    val_dataset = DatasetLucas(csv = args.val_csv, batch_size = args.batchsize, norm_type = args.norm_type, \
                                drop_last=False, vars=vars)
    # init tensorboard writer
    writer = SummaryWriter(args.experiment)
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
            # write on tensorboard
            writer.add_scalar('loss', cur_l, (ne*train_dataset.n_batches)+i)
            # print
            print('[{:d}/{:d}] Loss: {:.4f} - Val: {:.4f}' \
                .format(
                ne, args.nepochs, cur_l,
                best_val.loc['avg'].value if best_val is not None else 0.0))
            # backward propagation
            cur_l.backward()
            # update weights
            optimizer.step()
        # test on validation dataset
        df = test(net, val_dataset)
        print('\n',df,'\n')
        # save on tensorboard
        for col in df.index:
            writer.add_scalar(col, df.loc[col].value, ne)
        # create state
        state = {'net':net.state_dict(), 'opt':optimizer.state_dict(), 'vars':vars, 'res':df}
        torch.save(state, os.path.join(args.experiment,'latest.pth'))
        # update best value
        if best_val is None or df.loc['avg'].value < best_val.loc['avg'].value:
            torch.save(state, os.path.join(args.experiment, 'best.pth'))
            best_val = df
        # save the best value obtained on tensorboard
        writer.add_scalar('best', best_val.loc['avg'].value, ne)
