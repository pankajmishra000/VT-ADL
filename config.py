# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:26:06 2019

@author: piccia
"""
import torch

class Config():
    def __init__(self):
        self.USE_CUDA = True and torch.cuda.is_available()
    

        