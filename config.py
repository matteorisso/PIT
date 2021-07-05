# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:25:27 2020

@author: MatteoRisso
"""
#
# PARAMETERS : BEFORE RUNNING TAKE CARE THAT ARE THE CORRECT ONES
#

class Config:
    def __init__(self, dataset):
        self.dataset = dataset 

        # Where data are saved
        self.saving_path = r'./saved_models/'
        
        if dataset == 'PPG_Dalia':
            # Data preprocessing parameters. Needs to be left unchanged
            self.time_window = 8
            self.input_shape = 32 * self.time_window
        
            # Training Parameters
            self.batch_size = 128
            self.lr = 0.001
            self.epochs = 500
            self.a = 35
        elif dataset == 'Nottingham':
            # Architecture parameters
            self.n_channels = [150] * 4
            self.k = 6
            self.dp = 0.2
            self.n_classes = 88
        
            # Training Parameters 
            self.batch_size = 128
            self.lr = 1e-4
            self.epochs = 100
            self.a = 35
        elif dataset == 'JSB_Chorales':
            # Architecture parameters
            self.n_channels = [150] * 2
            self.k = 3
            self.dp = 0.5
            self.n_classes = 88
            
            # Training Parameters 
            self.batch_size = 128
            self.lr = 1e-3
            self.epochs = 100
            self.a = 35
        elif dataset == 'SeqMNIST' or dataset == 'PerMNIST':
            # Architecture parameters
            self.n_channels = [25] * 8
            self.k = 7
            self.dp = 0.
            self.n_classes = 10
            
            # Training Parameters 
            self.batch_size = 64
            self.lr = 1e-3
            self.epochs = 70
            self.a = 35
        
        # dataset location
        self.path_PPG_Dalia = r'./Dataset/PPG_Dalia/S'
        self.path_Nottingham = r'./Dataset/polymusic/Nottingham.mat'
        self.path_JSB_Chorales = r'./Dataset/polymusic/JSB_Chorales.mat'
        
        # warmup_epochs determines the number of training epochs without regularization
        # it could be an integer number or the string 'max' to indicate that we fully train the 
        # network
        self.warmup = 20
        # reg_strength determines how agressive lasso-reg is
        self.reg_strength = 1e-6
        # Amount of l2 regularization to be applied. Usually 0.
        self.l2 = 0.
        # threshold value is the value at which a weight is treated as 0. 
        self.threshold = 0.5
        
        self.hyst = 0

