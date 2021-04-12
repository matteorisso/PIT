# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:25:27 2020

@author: MatteoRisso
"""


#
# PARAMETERS : BEFORE RUNNING TAKE CARE THAT ARE THE CORRECT ONES
#

# Where data are saved
# set machine to server or local_linux or local_windows or colab
machine = 'server'
if machine == 'server':
    saving_path = '/space/risso/saved_models/'
elif machine == 'sassauna':
    saving_path = '/usr/scratch/sassauna1/aburrello/NAS_dil/saved_models/'
elif machine == 'local_windows' :
    saving_path = r'D:/Admin/Documents/LibriUniversitari/GitHub/ppg-on-chip/saved_models/'
elif machine == 'local_linux':    
    saving_path = '/home/matteorisso/Documents/ppg-on-chip/saved_models/'
else:
    saving_path = '/content/saved_models/'

# dataset could be  : 
# PPG_Dalia 
# Nottingham
# JSB_Chorales
# SeqMNIST
# PerMNIST
# AddProb
# Word_PTB
# LAMBADA
# Char_PTB
dataset = 'PPG_Dalia'

if dataset == 'PPG_Dalia':
    # Time window : 8s, 4s, 2s, 1s
    time_window = 8
    input_shape = 32*time_window
    # parameters training
    batch_size = 128
    lr = 0.001
    epochs = 500
    a = 35

    original_ofmap = False

elif dataset == 'Nottingham':
    n_channels = [150] * 4
    k = 6
    dp = 0.2

    n_classes = 88
    
    batch_size = 32
    lr = 1e-4
    epochs = 100
    a = 35
elif dataset == 'JSB_Chorales':
    n_channels = [150] * 2
    k = 3
    dp = 0.5

    n_classes = 88
    
    batch_size = 32
    lr = 1e-3
    epochs = 100
    a = 35
elif dataset == 'SeqMNIST' or dataset == 'PerMNIST':
    n_channels = [25] * 8
    k = 7
    dp = 0.

    n_classes = 10
    
    batch_size = 64
    lr = 1e-3
    epochs = 70
    a = 35
elif dataset == 'AddProb':
    n_channels = [24] * 8
    k = 8
    dp = 0.
    
    T = 600
    n_classes = 2
    
    batch_size = 512
    lr = 4e-3
    epochs = 30
    a = 35
    
    N_train = 50000
    N_test = 10000
elif dataset == 'Word_PTB':
    n_channels = [600] * 4
    k = 3
    dp = 0.5
    
    emb_size = 600
    validseqlen = 40
    seqlen = 80    
    
    batch_size = 16
    lr = 1e-3
    epochs = 500 
    a = 35
elif dataset == 'LAMBADA':
    n_channels = [500] * 5
    k = 4
    dp = 0.4
    
    emb_size = 500
    validseqlen = 50
    seqlen = 100    
    
    batch_size = 20
    lr = 1e-1
    epochs = 100
    a = 35
elif dataset == 'Char_PTB':
    n_channels = [450] * 3
    k = 3
    dp = 0.1
    
    emb_size = 100
    validseqlen = 320
    seqlen = 400
    
    batch_size = 32
    lr = 1e-2
    epochs = 100
    a = 35
    
# dataset location
if machine=='server':
    path_PPG_Dalia = r'/space/risso/PPG_Dalia/PPG_FieldStudy/S'
    path_IEEE_Training = r'/space/risso/competition_data/'
    path_Nottingham = r'/space/risso/polymusic/Nottingham.mat'
    path_JSB_Chorales = r'/space/risso/polymusic/JSB_Chorales.mat'
elif machine == 'local_windows' :
    path_PPG_Dalia = r'D:/Admin/Documents/LibriUniversitari/GitHub/ppg-on-chip/PPG_Dalia/S'
elif machine == 'sassauna':
    path_PPG_Dalia = '/usr/scratch/sassauna1/aburrello/PPG_Dalia/data/1Q16/PPG_FieldStudy/S'
elif machine == 'local_linux':    
    path_PPG_Dalia = '/home/matteorisso/Desktop/ppg_on_chip_val/PPG_FieldStudy/S'
else :
    path_PPG_Dalia = '/content/PPG_FieldStudy/S'


# warmup_epochs determines the number of training epochs without regularization
# it could be an integer number or the string 'max' to indicate that we fully train the 
# network
warmup = 20
# reg_strength determines how agressive lasso-reg is
l2 = 0.
reg_strength = 1e-6
# threshold value is the value at which a weight is treated as 0. 
threshold = 0.5

hyst = 0
epsilon = 0.01
