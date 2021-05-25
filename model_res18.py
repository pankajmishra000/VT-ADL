# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:14:10 2020

@author: Pankaj Mishra
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from config import Config

    
## Decoder ##
   
class decoder2(nn.Module):
    def __init__(self, in_channels):
        super(decoder2, self).__init__()
        self.decoder2 = nn.Sequential(
             nn.ConvTranspose2d(in_channels= in_channels, out_channels=16,kernel_size= 3, stride=2,padding=1),  # In b, 8, 8, 8 >> out b, 16, 15, 15
             nn.BatchNorm2d(16, affine = True),
             nn.ReLU(True),            
             nn.ConvTranspose2d(16, 32, 9, stride=3, padding = 1),  #out> b,32, 49, 49
             nn.BatchNorm2d(32, affine = True),
             nn.ReLU(True),             
             nn.ConvTranspose2d(32, 32, 7, stride=5, padding=1),  #out> b, 32, 245, 245
             nn.BatchNorm2d(32, affine = True),
             nn.ReLU(True), 
             nn.ConvTranspose2d(32, 16, 9, stride=2),  #out> b, 16, 497, 497
             nn.BatchNorm2d(16, affine = True),
             nn.ReLU(True), 
             nn.ConvTranspose2d(16, 8, 6, stride=1),  #out> b, 8, 502, 502
             nn.BatchNorm2d(8, affine = True),
             nn.ReLU(True),
             nn.ConvTranspose2d(8, 3, 11, stride=1),  #out> b, 3, 512, 512
             nn.Tanh()
             )
        
    def forward(self, x):
         recon = self.decoder2(x)
         return recon
                  
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


        
if __name__=="__main__":
    from torchsummary import summary
#    mod = Res18().cuda()    
#    summary(mod, input_size=(3,64,64))
#    
    decod = decoder1().cuda()
    summary(decod, input_size=(64,512))
#    
    # decod2 = decoder2(8).cuda()
    # summary(decod2, input_size=(8,8,8))
    # model = teacher_cap().cuda()
    # summary(model, input_size=(3,512,512))
