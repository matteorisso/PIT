# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:55:50 2020

@author: MatteoRisso
"""

import numpy as np
import config as cf
import RandomGroupkfold as rgkf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils import shuffle
import pickle
import json
import build_TEMPONet

def warmup(model, epochs_num, X_sh, y_sh, early_stop, checkpoint):
    hist = model.fit(x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), \
                        y=y_sh, shuffle=True, \
                        validation_split=0.1, verbose=1, \
                        batch_size= cf.batch_size, epochs=epochs_num,
                        callbacks = [early_stop, checkpoint])
    return

def train_gammas(model, X_sh, y_sh, early_stop, save_gamma, exp_str): 
    hist = model.fit(x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), \
                        y=y_sh, shuffle=True, \
                        validation_split=0.1, verbose=1, \
                        batch_size= cf.batch_size, epochs=cf.epochs,
                        callbacks = [early_stop, save_gamma, exp_str])
    return

def retrain(groups, X, y, activity, checkpoint, early_stop, ofmap):
    
    predictions = dict()
    MAE = dict()
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)

    # retrain and cross-validate
    result = rgkf.RandomGroupKFold_split(groups,4,cf.a)
    for train_index, test_val_index in result:
        X_train, X_val_test = X[train_index], X[test_val_index]
        y_train, y_val_test = y[train_index], y[test_val_index]
        activity_train, activity_val_test = activity[train_index], activity[test_val_index]

        logo = LeaveOneGroupOut()
        logo.get_n_splits(groups=groups[test_val_index])  # 'groups' is always required
        for validate_index, test_index in logo.split(X_val_test, y_val_test, groups[test_val_index]):
            X_validate, X_test = X_val_test[validate_index], X_val_test[test_index]
            y_validate, y_test = y_val_test[validate_index], y_val_test[test_index]
            activity_validate, activity_test = activity_val_test[validate_index], activity_val_test[test_index]
            groups_val=groups[test_val_index]
            k=groups_val[test_index][0]
        
            # init
            try:
               del model
            except:
               pass

            # obtain conv #output filters from learned json structure
            with open(cf.saving_path+'autodil/learned_dil_'+'{:.1e}'.format(cf.reg_strength)+'_'+'{:.1e}'.format(cf.threshold)+'_{}'.format(cf.warmup)+'.json', 'r') as f:
                dil_list = [val for _,val in json.loads(f.read()).items()]

            model = build_TEMPONet.TEMPONet_learned(1, cf.input_shape, dil_list, ofmap=ofmap)

            # save model and weights
            val_mae = 'val_mean_absolute_error'
            mae = 'mean_absolute_error'
            checkpoint = ModelCheckpoint(cf.saving_path+'test_reg'+str(k)+'.h5', monitor=val_mae, verbose=1,\
            save_best_only=True, save_weights_only=False, mode='min', period=1)
            #configure  model
            adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            model.compile(loss='logcosh', optimizer=adam, metrics=[mae])


            X_train, y_train = shuffle(X_train, y_train)
            print(X_train.shape)
            print(X_validate.shape)
            print(X_test.shape)

            # Training
            hist = model.fit(x=np.transpose(X_train.reshape(X_train.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y=y_train, epochs=cf.epochs, batch_size=cf.batch_size, \
           validation_data=(np.transpose(X_validate.reshape(X_validate.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), y_validate), verbose=1,\
           callbacks=[checkpoint, early_stop])

            #evaluate
            predictions[k] = model.predict(np.transpose(X_test.reshape(X_test.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)))
            MAE[k] = np.linalg.norm(y_test-predictions[k], ord=1)/y_test.shape[0]

            print(MAE)

    return model, MAE
