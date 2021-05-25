# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:19:16 2021

@author: pankaj.mishra
"""
from scipy.ndimage import gaussian_filter, median_filter
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import label


def Normalise(score_map):
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    return scores
def Mean_var(score_map):
    mean = np.mean(score_map)
    var = np.var(score_map)
    return mean, var
def Filter(score_map, type=0):
    '''
    Parameters
    ----------
    score_map : score map as tensor or ndarray
    type : Int, optional
            DESCRIPTION. The values are:
            0 = Gaussian
            1 = Median

    Returns
    -------
    score: Filtered score

    '''
    if type ==0:
        score = gaussian_filter(score_map, sigma=4)
    if type == 1:
        score = median_filter(score_map, size=3)
    return score

def Binarization(mask, thres = 0., type = 0):
    if type == 0:
        mask = np.where(mask > thres, 1., 0.)
    elif type ==1:
        mask = np.where(mask > thres, mask, 0.)
    return mask  

def plot(image,grnd_truth, score):
    plt.subplot(131)
    plt.imshow(image[0].permute(1,2,0))
    plt.subplot(132)
    plt.imshow(grnd_truth.squeeze(0).squeeze(0))
    plt.xlabel('ground truth')
    plt.subplot(133)
    plt.imshow(score)
    plt.xlabel('predicted')
    # plt.title('Anomaly score')
    # plt.imshow(score[0].permute(1,2,0), cmap='Reds')
    plt.colorbar()
    plt.pause(1) 
    plt.show()          
def binImage(heatmap, thres=0 ):
    _, heatmap_bin = cv2.threshold(heatmap , thres , 255 , cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin
def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
       lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc 