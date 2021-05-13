import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class Net(nn.Module):
    def __init__(self, nemb=12, nch=1, powf=3, max_powf=8, insz=1024, minsz=8, nbsr=1, leak=0.2, \
                batch_momentum=.01, use_batchnorm=True, sigmoid_from=None):
        '''
        @param nemb size of the embedding
        @param nch number of input channels
        @param powf 2^nch is the number of filters in LAST layer before projection
        @param insz size of the input tensor
        @param minsz minimum size of the map before calculating embedding
        '''
        super(Net, self).__init__()
        # save attributes
        self.nb = nbsr
        self.insz = insz
        self.leak = leak
        self.batch_momentum = batch_momentum
        self.sigmoid_from = sigmoid_from
        # calculate number of blocks to arrive to 1 x minsz
        nblocks = int(math.log(float(insz)/float(minsz), 2))
        # for each block define the number of filters
        self.n_filters = [nch] + [2**min((i+powf),max_powf) for i in range(nblocks)]
        # print number of filters
        print('The number of filters are: [{}]'.format(','.join([str(f) for f in self.n_filters])))
        # create layers
        down_blocks = []
        for i in range(len(self.n_filters)-1):
            down_blocks.extend(self.down_block( self.n_filters[i], self.n_filters[i+1], use_batchnorm ))
        self.down = nn.Sequential(*down_blocks)
        # create embedding
        cur_sz = int(insz / (2**(len(self.n_filters)-1)))
        inter_ch =  int(abs(self.n_filters[-1]+nemb)/2)
        emb = []
        emb.append(nn.Conv1d(self.n_filters[-1], inter_ch, kernel_size=cur_sz, stride=cur_sz, padding=0))
        if use_batchnorm:
            emb.append(nn.BatchNorm1d(inter_ch, momentum=self.batch_momentum))
        if self.leak > 0:
            emb.append(nn.LeakyReLU(negative_slope=self.leak, inplace=True))
        else:
            emb.append(nn.ReLU(inplace=True))
        emb.append(nn.Conv1d(inter_ch, nemb, kernel_size=1, stride=1, padding=0))
        self.emb = nn.Sequential(*emb)


    def forward(self, x):
        if x.ndim==4: x = x.squeeze(1)
        x = F.interpolate(x, size=self.insz, mode='linear', align_corners=True)
        x = self.down(x)
        x = self.emb(x)
        x = x.squeeze(-1)
        if self.sigmoid_from:
            x[:,self.sigmoid_from:] = torch.sigmoid(x[:,self.sigmoid_from:])
        return x


    def down_block(self, inch, outch, use_batchnorm):
        blocks = []
        # create down block
        blocks.append(nn.Conv1d(inch, outch, kernel_size=4, stride=2, padding=1))
        if use_batchnorm:
            blocks.append(nn.BatchNorm1d(outch, momentum=self.batch_momentum))
        if self.leak > 0:
            blocks.append(nn.LeakyReLU(negative_slope=self.leak, inplace=True))
        else:
            blocks.append(nn.ReLU(inplace=True))

        # add same resolution blocks
        for i in range(self.nb):
            blocks.append(nn.Conv1d(outch, outch, kernel_size=3, stride=1, padding=1))
            if use_batchnorm:
                blocks.append(nn.BatchNorm1d(outch, momentum=self.batch_momentum))
            if self.leak > 0:
                blocks.append(nn.LeakyReLU(negative_slope=self.leak, inplace=True))
            else:
                blocks.append(nn.ReLU(inplace=True))

        # return blocks
        return blocks


if __name__ == '__main__':
    # init network
    emb = Net(nemb=12, nch=1, powf=4, max_powf=7, insz=4096, minsz=4, nbsr=0, leak=0.2, batch_momentum=.01, sigmoid_from=10)
            # stride, avgpool maxpool
    emb.cuda()
    # describe
    from torchsummary import summary
    summary(emb, (1, 4200))
    # print output size
    print(emb(torch.rand(10,1,1,4200).cuda()).size())
