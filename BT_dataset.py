#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:05:43 2020

@author: riccardo
"""

from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import numpy as np
import torch
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image
# import cv2
import pandas as pd

def show_img(x):
    plt.imshow(x, cmap='gray')
    plt.show()

def load_images(path):
    return io.imread(path)

def load_masks(path):
    return io.imread(path,as_gray=True)


class Dataset(data.Dataset):
    def __init__(self, image_paths, labels,resize):
        self.paths = image_paths
        self.labels = labels
        self.resize = resize
       
        
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((300,5024)),
            transforms.ToTensor()])
        self.default_transform = transforms.Compose([transforms.ToTensor()]) 
        
        # self.default_transform = transforms.Compose([transforms.Resize((300,1023)),transforms.ToTensor(), transforms.Normalize((0.47919376, 0.47919376, 0.15294718), (0.02388213, 0.02388213, 0.1695453))]) #normalized for pretrained network
        # self.default_transform_mask = transforms.Compose([transforms.Resize((300,1023)),transforms.ToTensor()])
    def __len__(self):
        return self.paths.shape[0]
    
    def __getitem__(self, i):
        # image = np.load(self.paths[i]) #load from .npy file!
        image_ = load_images(self.paths[i]) #load from .png file!
        if(len(image_.shape)<3):
                image_ = np.stack((image_,)*3, axis=-1)
                
        if(self.resize):
            image = self.resize_transform(image_)
        else:
            image = self.default_transform(image_)
        
        # load masks
        # label = load_masks(self.labels[i])
        # label = self.default_transform_mask(label[:,:,None])
        label = torch.zeros(1,image.size(1), image.size(2))
        return image, label
    
    
    def loader(csv_name, BATCH_SIZE, resize):
        # LOAD DATASET PATHS
        train_csv = pd.read_csv(csv_name)
        train_paths = train_csv.CAM
        train_labels = None
        
        # SPLIT TRAIN - VALIDATION
        # train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_paths, train_labels, test_size = 0.33, random_state=23)#, stratify = train_labels)
        # train_paths.reset_index(drop=True,inplace=True)
        # train_labels.reset_index(drop=True,inplace=True)    
        # valid_paths.reset_index(drop=True,inplace=True)
        # valid_labels.reset_index(drop=True,inplace=True)
        
        # SPLIT VALIDATION -TEST
        # valid_paths, test_paths, valid_labels, test_labels = train_test_split(valid_paths, valid_labels, test_size = 0.33, random_state=23)#, stratify = train_labels)
        
        # valid_paths.reset_index(drop=True,inplace=True)
        # valid_labels.reset_index(drop=True,inplace=True)
        # test_paths.reset_index(drop=True,inplace=True)
        # test_labels.reset_index(drop=True,inplace=True)
        
        # TRAIN LOADER
        train_dataset = Dataset(train_paths, train_labels, resize)
        train_loader =  torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size = BATCH_SIZE)#, num_workers = 2)
        
        # VALIDATION LOADER
        # valid_dataset = BTDataset(valid_paths, valid_labels, color_space)
        # validation_loader =  torch.utils.data.DataLoader(valid_dataset, shuffle=False, batch_size = BATCH_SIZE)#, num_workers = 2)
        
        # TEST LOADER
        # test_dataset = BTDataset(test_paths, test_labels, color_space)
        # test_loader =  torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size = BATCH_SIZE)#, num_workers = 2))
        
        return train_loader #, validation_loader, test_loader
       
