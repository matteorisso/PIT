import numpy as np
import tensorflow as tf

import preprocessing as pp
import config as cf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils import shuffle


import build_TEMPONet

import utils

val_mae = 'val_mean_absolute_error'
mae = 'mean_absolute_error'

early_stop = EarlyStopping(monitor=val_mae, min_delta=0.01, patience=35, mode='min', verbose=1)

# Load data
X, y, groups, activity = pp.preprocessing(cf.dataset)

# organize data
group_kfold = GroupKFold(n_splits=4)
group_kfold.get_n_splits(X, y, groups)

predictions = dict()
MAE = dict()
dataset = dict()

# Learn dil fact
model = build_TEMPONet.TEMPONet_d1(1, cf.input_shape)
del model
model = build_TEMPONet.TEMPONet_d1(1, cf.input_shape, trainable=False)

# save model and weights
checkpoint = ModelCheckpoint(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5', monitor=val_mae, verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
#configure  model
adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='logcosh', optimizer=adam, metrics=[mae])

X_sh, y_sh = shuffle(X, y)

print('Train model for {} epochs'.format(cf.warmup))
strg = cf.reg_strength
cf.reg_strength = 0

if cf.warmup == 'max':
    epochs_num = cf.epochs
else:
    epochs_num = cf.warmup

hist = model.fit(x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1),                         (0, 3, 2, 1)), \
                         y=y_sh, shuffle=True, \
                         validation_split=0.1, verbose=1, \
                         batch_size= cf.batch_size, epochs=epochs_num,
                         callbacks = [early_stop, checkpoint])
cf.reg_strength = strg
