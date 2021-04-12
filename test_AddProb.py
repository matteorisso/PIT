# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:00:40 2020

@author: MatteoRisso
"""

import pdb
import tensorflow as tf
import numpy as np

from tensorflow.keras.optimizers import Adam


from sklearn.utils import shuffle

import config as cf 

import tensorflow.keras.backend as K

import build_ResTCN
import train_ResTCN

# Generate data
np.random.seed(cf.a)
# training set
X_num = np.random.uniform(0, 1, (cf.N_train, 1, cf.T))
X_mask = np.zeros([cf.N_train, 1, cf.T])
y_train = np.zeros([cf.N_train, 1])
for i in range(cf.N_train):
    positions = np.random.choice(cf.T, size=2, replace=False)
    X_mask[i, 0, positions[0]] = 1
    X_mask[i, 0, positions[1]] = 1
    y_train[i,0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
X_train = np.concatenate((X_num, X_mask), axis=1)

# test set
X_num = np.random.uniform(0, 1, (cf.N_test, 1, cf.T))
X_mask = np.zeros([cf.N_test, 1, cf.T])
y_test = np.zeros([cf.N_test, 1])
for i in range(cf.N_test):
    positions = np.random.choice(cf.T, size=2, replace=False)
    X_mask[i, 0, positions[0]] = 1
    X_mask[i, 0, positions[1]] = 1
    y_test[i,0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
X_test = np.concatenate((X_num, X_mask), axis=1)


X_train, y_train = shuffle(X_train, y_train)


# Build model
model = build_ResTCN.ResTCN(cf.n_classes, cf.n_channels, cf.k, cf.dp, variant='AddProb')

opt = Adam(lr=cf.lr, clipnorm=1.0)

model.compile(
             loss='mean_squared_error',
             optimizer=opt)

# Train
hist = model.fit(
    x=X_train.reshape(-1, 1, X_train.shape[-1], 2),
    y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
    validation_data=(X_test.reshape(-1, 1, X_test.shape[-1], 2), y_test.reshape(y_test.shape[0], 1)), 
    verbose=1, 
    batch_size= cf.batch_size, epochs=cf.epochs)