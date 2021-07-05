# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:49:10 2020

Preprocessing function for correctly managing data present in the IEEE_Training
dataset and/or the PPG_Dalia dataset.
It takes as input the desired dataset and returns:
    - X :the data to be used for training, validation and test purposes 
    - y : ground truth array
    - groups : specification of 

@author: matteorisso
"""

import pickle
import numpy as np
from skimage.util.shape import view_as_windows
from scipy.io import loadmat
import random
#import config as cf

def preprocessing(dataset, cf):
    # Sampling frequency of both ppg and acceleration data in IEEE_Training dataset
    fs_IEEE_Training = 125
    # Sampling frequency of acceleration data in PPG_Dalia dataset
    # The sampling frequency of ppg data in PPG_Dalia dataset is fs_PPG_Dalia*2
    fs_PPG_Dalia = 32
    
    fs_activity = 4
    
    Sessioni = dict()
    S = dict()
    acc = dict()
    ppg = dict()
    activity = dict()
    
    random.seed(20)
     
    ground_truth = dict()
    
    val = dataset

    numbers= list(range(1,16))
    session_list=random.sample(numbers,len(numbers))
    for j in session_list:
        paz = j
        
        with open(cf.path_PPG_Dalia + str(j) +'/S' + str(j) +'.pkl', 'rb') as f:
            S[paz] = pickle.load(f, encoding='latin1')
        ppg[paz] = S[paz]['signal']['wrist']['BVP'][::2]
        acc[paz] = S[paz]['signal']['wrist']['ACC']
        activity[paz] = S[paz]['activity']
        ground_truth[paz] = S[paz]['label']
        
    sig = dict()
    act_list = []
    groups= []
    sig_list = []
    ground_truth_list = []
    
    # Loop on keys of dictionary ground_truth
    for k in ground_truth:
        # Remeber to set the desired time window
        activity[k] = np.moveaxis(view_as_windows(activity[k], (4*cf.time_window,1),4*2)[:,0,:,:],1,2)
        activity[k] = activity[k][:,:,0]
        sig[k] = np.concatenate((ppg[k],acc[k]),axis=1)
        sig[k]= np.moveaxis(view_as_windows(sig[k], (fs_PPG_Dalia*cf.time_window,4),fs_PPG_Dalia*2)[:,0,:,:],1,2)
        groups.append(np.full(sig[k].shape[0],k))
        sig_list.append(sig[k])
        act_list.append(activity[k])
        ground_truth[k] = np.reshape(ground_truth[k], (ground_truth[k].shape[0],1))
        ground_truth_list.append(ground_truth[k])

    #print("gruppo",groups)
    groups = np.hstack(groups)
    X = np.vstack(sig_list)
    y = np.reshape(np.vstack(ground_truth_list),(-1,1))
    
    act = np.vstack(act_list)
   
    print("dimensione train",X.shape, "dimesione test", y.shape,"dimensione gruppi",groups.shape)
    
    return X[:y.shape[0]], y, groups[:y.shape[0]], act[:y.shape[0]]
