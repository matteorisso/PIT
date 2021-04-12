# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 12:39:29 2020

@author: MatteoRisso
"""

import pdb
import tensorflow as tf
import numpy as np

import config as cf

from tensorflow.keras import backend as K

# Limit GPU usage
if cf.machine == 'server':
    if tf.__version__ == '1.14.0':
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # some aliases necessary in tf 1.14
        val_mae = 'val_mean_absolute_error'
        mae = 'mean_absolute_error'
    else:
        limit = 1024 * 5
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
          try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=limit)])
          except RuntimeError as e:
            print(e)
 
from scipy.io import loadmat
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.utils import shuffle

from train_ResTCN import train_Nottingham

import build_ResTCN

#class NLL(tf.keras.losses.Loss):
@tf.function
def NLL(y_true, y_pred):
    #print("y_true :", tf.print(y_true))
    #print("y_pred :", tf.print(y_pred))
    #tf.print(tf.math.is_nan(tf.cast(y_true, dtype='float32')))
    #tf.print(tf.math.is_nan(y_pred))
    return -tf.linalg.trace(
        tf.matmul(
            tf.cast(y_true, dtype='float32'),
            tf.transpose(tf.cast(tf.math.log(
                tf.clip_by_value(y_pred, 1e-10, 1.)), dtype='float32'), [0, 1, 3, 2])
            ) +
        tf.matmul(
            tf.cast((1 - y_true), dtype='float32'),
            tf.transpose(tf.cast(tf.math.log(
                1 - tf.clip_by_value(y_pred, 0., 1-1e-10)), dtype='float32'), [0, 1, 3, 2])
            )
        )
    
#
# load data
#

#JSB_Chorales or Nottingham
data = loadmat('./old/TCN/polymusic/JSB_Chorales.mat')

X_train = data['traindata'][0]
X_valid = data['validdata'][0]
X_test = data['testdata'][0]

model = build_ResTCN.ResTCN(cf.n_classes, cf.n_channels, cf.k, cf.dp, variant='Nottingham')

learning_rate_fn = tf.keras.optimizers.schedules.InverseTimeDecay(
    cf.lr, 1e3, 9, staircase=True)
opt = Adam(learning_rate=cf.lr, clipnorm=0.4)
#opt = RMSprop(lr=lr, clipvalue=1.0)

model.compile(
        #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        loss=NLL,
        optimizer=opt) 
        #metrics=[mae])

train_losses = list()
valid_losses = list()
test_losses = list()
epoch = 0

lr = cf.lr
for ep in range(cf.epochs):
    
    if ep > 5:
        lr /= 10
        K.set_value(model.optimizer.lr, K.get_value(lr))
    
    print(K.get_value(lr))
    
    tr_l, v_l, t_l = train_Nottingham(model, ep, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses)
    
    
    
    
    

 


