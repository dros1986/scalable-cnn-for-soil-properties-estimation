import torch
import torch.nn as nn
import torch.nn.functional as F




# define network
class Net(nn.Module):
    def __init__(self, dropout=0, momentum=0.01):
        super(Net, self).__init__()
        self.momentum = momentum
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
            nn.BatchNorm2d(64, momentum=self.momentum),
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
    		nn.BatchNorm2d(ch_out, momentum=self.momentum),
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
