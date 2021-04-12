# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:07:23 2020

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

from preprocessing_LAMBADA import data_generator, batchify, get_batch

#from train_ResTCN import train_LAMBADA

#
# load data
#

train_data, val_data, test_data, corpus = data_generator('LAMBADA', cf.seqlen)

n_words = len(corpus.dictionary)

n_classes = list()
n_classes.append(n_words)
n_classes.append(cf.emb_size)

num_chans = cf.n_channels[:-1] + [cf.emb_size]

model = build_ResTCN.ResTCN(n_classes, num_chans, cf.k, cf.dp, variant='LAMBADA')
