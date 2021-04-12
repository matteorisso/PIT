# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:05:42 2020

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

def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())

# Load data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize
mean = 0.1307
std = 0.3081

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
#X_train = X_train // 255
#X_test = X_test // 255

# Serialize
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]*X_train.shape[2]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]*X_test.shape[2]))

permute = 1
if permute == 1:
    perm = np.random.permutation(X_train.shape[-1])
    X_train = X_train[:,:,perm]
    X_test = X_test[:,:,perm]

X_train, y_train = shuffle(X_train, y_train)

n_batch = X_train.shape[0] // cf.batch_size

# Build model
#model = build_ResTCN.ResTCN_d1(cf.n_classes, cf.n_channels, cf.k, cf.dp, variant='SeqMNIST')
model = build_ResTCN.ResTCN(cf.n_classes, cf.n_channels, cf.k, cf.dp, variant='SeqMNIST')

opt = Adam(lr=cf.lr)

model.compile(
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
             optimizer=opt,
             metrics=['accuracy'])

# Train
hist = model.fit(
    x=X_train.reshape(-1, 1, X_train.shape[-1], 1),
    y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
    validation_data=(X_test.reshape(-1, 1, X_test.shape[-1], 1), y_test.reshape(y_test.shape[0], 1)), 
    verbose=1, 
    batch_size= cf.batch_size, epochs=cf.epochs)
'''
epoch_losses = list()
epoch_accuracy = list()
epoch_losses_test = list()
for ep in range(cf.epochs):
    batch_loss = list()
    batch_accuracy = list()
    
    for batch in range(n_batch+1):
        
        x_batch_train = X_train[(batch*cf.batch_size):((batch+1)*cf.batch_size),:,:]
        y_batch_train = y_train[(batch*cf.batch_size):((batch+1)*cf.batch_size)]
        y_batch_train = y_batch_train.reshape((y_batch_train.shape[0], 1))
        
        loss, accuracy = model.train_on_batch(
           x = x_batch_train.reshape((-1, 1, x_batch_train.shape[-1], 1)),
           y = y_batch_train)
        #y = tf.reshape(y_batch_train, (cf.n_classes)))
           
        batch_loss.append(loss)
        batch_accuracy.append(accuracy)
        
    epoch_losses.append(sum(batch_loss) / len(batch_loss))
    epoch_accuracy.append(sum(batch_accuracy) / len(batch_accuracy))
    print("Epoch : {e} \t Loss : {l} \t Acc : {a}".format(e=ep, l=epoch_losses[ep], a=epoch_accuracy[ep]))
'''













