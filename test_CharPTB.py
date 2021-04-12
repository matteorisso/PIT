# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 21:51:42 2020

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

import build_ResTCN

from preprocessing_CharPTB import data_generator, batchify, get_batch, char_tensor

from train_ResTCN import train_Char_PTB

#tf.compat.v1.disable_eager_execution()


#
# load data
#

file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator('Char_PTB')

eval_batch_size = 10
X_train = batchify(char_tensor(corpus, file), cf.batch_size)
X_valid = batchify(char_tensor(corpus, valfile), 1)
X_test = batchify(char_tensor(corpus, testfile), 1)

n_chars = len(corpus.dict)

n_classes = list()
n_classes.append(n_chars)
n_classes.append(cf.emb_size)

num_chans = cf.n_channels[:-1] + [cf.emb_size]

model = build_ResTCN.ResTCN(n_classes, num_chans, cf.k, cf.dp, variant='Char_PTB')

opt = SGD(learning_rate=cf.lr, clipvalue=0.15)
#opt = Adam(learning_rate=1e-3, clipnorm=0.4)
# model.compile(
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         optimizer=opt) 

train_losses = list()
valid_losses = list()
test_losses = list()
epoch = 0

lr = cf.lr
for ep in range(cf.epochs):
    
    tr_l, v_l, t_l = train_Char_PTB(model, ep, n_chars, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses, lr)
    
    if ep > 5 and v_l >= max(valid_losses[-5:]):
        lr /= 2.
         #K.set_value(model.optimizer.lr, K.get_value(lr))
    print(K.get_value(lr))