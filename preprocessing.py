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
import config as cf

def preprocessing(dataset):
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

    if val=='IEEE_Training':
        numbers= list(range(1,13))
        session_list=random.sample(numbers,len(numbers))
        for j in session_list:
            paz = j
            if j<10:
                Sessioni[paz] = np.moveaxis(loadmat(cf.path_IEEE_Training+'DATA_0'+ str(j) +  '_TYPE02'  + '.mat')['sig'],0,1)
                ground_truth[paz] = loadmat(cf.path_IEEE_Training+'DATA_0'+ str(j) +  '_TYPE02_BPMtrace'  + '.mat')['BPM0']
            else:
                Sessioni[paz] = np.moveaxis(loadmat(cf.path_IEEE_Training+'DATA_'+ str(j) +  '_TYPE02'  + '.mat')['sig'],0,1)
                ground_truth[paz] = loadmat(cf.path_IEEE_Training+'DATA_'+ str(j) +  '_TYPE02_BPMtrace'  + '.mat')['BPM0']
    
    elif val=='PPG_Dalia':
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
            
    
    else:
         numbers= list(range(1,10))
         session_list=random.sample(numbers,len(numbers))
         for j in session_list:
            paz = j
            if j<10:
                Sessioni[paz] = np.moveaxis(loadmat('data/testset/TestData/' +'TEST_S0'+ str(j) + '.mat')['sig'],0,1)
                ground_truth[paz] = loadmat('data/testset/TrueBPM/' +'True_S0'+ str(j)   + '.mat')['BPM0']
            else:
                Sessioni[paz] = np.moveaxis(loadmat('data/testset/TestData/' +'TEST_S'+ str(j) + '.mat')['sig'],0,1)
                ground_truth[paz] = loadmat('data/testset/TrueBPM/' +'True_S'+ str(j)  + '.mat')['BPM0']
    
    
    sig = dict()
    act_list = []
    groups= []
    sig_list = []
    ground_truth_list = []
    
    if val=='IEEE_Training':
        # Loop on keys of dictionary Sessioni
        for k in Sessioni:
            #S=[]
            # Take the only data needed to feed the network,
            # ie : ppg, acc_x, acc_y, acc_z
            sig[k] = Sessioni[k][:,2::]
            sig[k]=np.moveaxis(view_as_windows(sig[k], (fs_IEEE_Training*8,4),fs_IEEE_Training*2)[:,0,:,:],1,2)
            groups.append(np.full(sig[k].shape[0],k))
            sig_list.append(sig[k])
            ground_truth_list.append(ground_truth[k])
    elif val=='PPG_Dalia':
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
    
    if val=='PPG_Dalia':
        act = np.vstack(act_list)
    
    #print(X.shape)
    #print(groups.shape)
    if val=='IEEE_Training':
        X = np.concatenate((X[:,:,::4],np.random.randint(3, size=(1768,4,6))), axis=2)
    
    #print(groups,groups.shape)
    print("dimensione train",X.shape, "dimesione test", y.shape,"dimensione gruppi",groups.shape)
    
    if val=='PPG_Dalia':
        return X[:y.shape[0]], y, groups[:y.shape[0]], act[:y.shape[0]]
    else:
        return X, y, groups
