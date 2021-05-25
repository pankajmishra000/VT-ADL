# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 15:40:29 2020

@author: Pankaj Mishra
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from config import Config


class Unity(nn.Module):
    def __init__(self,ks, in_ch=512):
        super(Unity,self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=512, kernel_size=ks)
        
    def forward(self,x):
        return F.relu(self.conv(x), inplace= True)
    
class Spatial_Scorer(nn.Module):
    def __init__(self,in_dim=512,test=False):
        super(Spatial_Scorer,self).__init__()
        self.test = test
        
        self.layers = nn.Sequential( nn.Linear(in_dim,256),
                                    nn.ReLU(True),
                                    nn.Linear(256,128),
                                    nn.ReLU(True),
                                    nn.Linear(128,1),
                                    nn.Tanh())
        if not self.test:
            print("Initializing Spatial socrer network.........")
            initialize_weights(self.layers)
        
    def forward(self, x):
        x = x.view(x.size(0),-1)
        F = self.layers(x)
        return F
        
# Initialize weight function
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
class DigitCaps(nn.Module):
    def __init__(self, out_num_caps=1, in_num_caps=8*8*64, in_dim_caps=8, out_dim_caps=512, decode_idx=-1):
        super(DigitCaps, self).__init__()

        self.conf = Config()
        self.in_dim_caps = in_dim_caps
        self.in_num_caps = in_num_caps
        self.out_dim_caps = out_dim_caps
        self.out_num_caps = out_num_caps
        self.decode_idx = decode_idx
        self.W = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
#        self.upsample = upsampling()

    def forward(self, x):
        # x size: batch x 1152 x 8
        x_hat = torch.squeeze(torch.matmul(self.W, x[:, None, :, :, None]), dim=-1)
        x_hat_detached = x_hat.detach()
        # x_hat size: batch x ndigits x 1152 x 16
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps))
        # b size: batch x ndigits x 1152
        if self.conf.USE_CUDA:
            b = b.cuda()

        # routing algo taken from https://github.com/XifengGuo/CapsNet-Pytorch/blob/master/capsulelayers.py
        num_iters = 3
        for i in range(num_iters):
            c = F.softmax(b, dim=1)
            # c size: batch x ndigits x 1152
            if i == num_iters -1:
                # output size: batch x ndigits x 1 x 16
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
            else:
                outputs = self.squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)


        outputs = torch.squeeze(outputs, dim=-2) # squeezing to remove ones at the dimension -1
        # Below code chooses the maximum lenth of the vector
        if self.decode_idx == -1:  # choose the longest vector as the one to decode
            classes = torch.sqrt((outputs ** 2).sum(2))
            classes = F.softmax(classes, dim=1)
            _, max_length_indices = classes.max(dim=1)
        else:  # always choose the same digitcaps
            max_length_indices = torch.ones(outputs.size(0)).long() * self.decode_idx
            if self.conf.USE_CUDA:
                max_length_indices = max_length_indices.cuda()

        masked = Variable(torch.sparse.torch.eye(self.out_num_caps))
        if self.conf.USE_CUDA:
            masked = masked.cuda()
        masked = masked.index_select(dim=0, index=max_length_indices)
#        t = (outputs * masked[:, :, None]).view(x.size(0), -1)
        t = (outputs * masked[:, :, None]).sum(dim=1).unsqueeze(1)
#        t = self.upsample(t)

        return t, outputs

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm *  input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor

    
    
if __name__=="__main__":
    from torchsummary import summary
    mod = Unity(16).cuda()
    print(mod)
    summary(mod, input_size=(512,16,16))
    
    model = Spatial_Scorer().cuda()
    x = torch.rand((225,512)).cuda()
    # print(model(x))
    summary(model, (1,512), batch_size=225)