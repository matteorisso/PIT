# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:11:50 2020

@author: MatteoRisso
"""
import pdb
import os
import pickle
import re
import sys
import numpy as np
import tensorflow as tf
import argparse
import config as cf
import subprocess

# Limit GPU usage
if cf.machine == 'server':
    if tf.__version__ == '1.14.0':
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
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
        # some aliases necessary in tf > 1.14
        val_mae = 'val_mae'
        mae = 'mae'

if tf.__version__ == '1.14.0':
    # some aliases necessary in tf 1.14
    val_mae = 'val_mean_absolute_error'
    mae = 'mean_absolute_error'
else:
    # some aliases necessary in tf > 1.14
    val_mae = 'val_mae'
    mae = 'mae'

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils import shuffle

from scipy.io import loadmat

import RandomGroupkfold as rgkf
import preprocessing as pp
import build_model
import config as cf
import utils
import json
import datetime
import re
from TCN.copy_memory import data_generator
from TCN import build_model as build_model_tcn

import matplotlib.pyplot as plt

class WarmUp(tf.keras.callbacks.Callback):
    def __init__(self, patience=10, strength=cf.reg_strength, l2=cf.l2):
        super(WarmUp, self).__init__()
        self.patience = patience
        self.strength = strength
        self.l2 = l2
            
    def on_train_begin(self, logs=None):
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        self.wait += 1
        if np.less(self.wait, self.patience):
            cf.reg_strength = 0
            cf.l2 = 0
            print("Still warm up for {} the strength is {}.".format(self.patience-self.wait, cf.reg_strength))

        else:
            cf.reg_strength = self.strength
            cf.l2 = self.l2
            print("Warm-up ended, strength is {}".format(cf.reg_strength))

class strength_control(tf.keras.callbacks.Callback):
    def __init__(self, delta=10):
        super(strength_control, self).__init__()
        self.delta = delta

    def on_train_begin(self, logs=None):
        self.best = np.Inf
        self.cnt = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_mean_absolute_error")

        if np.less(current, self.best):
            self.best = current
            self.cnt = 0
            self.nchange = 1
        else:
            self.cnt += 1

        if self.cnt == self.delta:
            cf.reg_strength = cf.reg_strength * 50/self.nchange
            print('New reg strength : {}'.format(cf.reg_strength))
            self.nchange += 1
            self.cnt = 0

class export_structure(tf.keras.callbacks.Callback):
    def __init__(self):
        super(export_structure, self).__init__()
        
    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = np.Inf
        self.gamma = dict()

    def on_epoch_end(self, epoch, logs=None):
        #get current validation mae
        if cf.dataset == 'PPG_Dalia':
            current = logs.get("val_mean_absolute_error")
        elif cf.dataset == 'copy_memory':
            current = logs.get("val_loss")
        elif cf.dataset == 'poly_music':
            current = logs.get("val_loss")

        #compare with previous best one
        if np.less(current, self.best):
            self.best = current
            
            # Record the best model if current results is better (less).
            names = [weight.name for layer in model.layers for weight in layer.weights]
            weights = model.get_weights()
            for name, weight in zip(names, weights):
                if re.search('learned_conv2d.+_?[0-9]/gamma', name):
                    self.gamma[name] = weight
                    self.gamma[name] = np.array(self.gamma[name] > cf.threshold, dtype=bool)
                    self.gamma[name] = utils.dil_fact(self.gamma[name], op='mul')
            
            print("New best MAE, update file. \n")    
            print(self.gamma) 
            utils.save_dil_fact(cf.saving_path, self.gamma)


class SaveGamma(tf.keras.callbacks.Callback):
   
    '''
    def on_batch_begin(self, batch, logs):
        print('Batch begin:')
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()

        i = 0
        for name, weight in zip(names, weights):
           if re.search('learned_conv2d.+_?[0-9]/alpha', name):
                print('alpha: ', weight.tolist()[-1])



    def on_batch_end(self, batch, logs):
        print('Batch end:')
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()

        i = 0
        for name, weight in zip(names, weights):
           if re.search('learned_conv2d.+_?[0-9]/alpha', name):
                print('alpha: ', weight.tolist()[-1])
'''


    def on_epoch_end(self, epoch, logs):
        
        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()

        gamma = dict()
        i = 0
        for name, weight in zip(names, weights):
            if re.search('learned_conv2d.+_?[0-9]/gamma', name):
                print('gamma: ', weight.tolist()[0])
                gamma[i] = weight.tolist()[0]
                #gamma[i] = np.array(gamma[i], dtype=bool)
                #gamma[i] = utils.dil_fact(gamma[i])
                i += 1
            if re.search('alpha.+_?[0-9]', name):
                print('alpha: ', weight.tolist()[-1])

        gamma_history.append(gamma)

def NLL(y_true, y_pred):
    
    return -tf.linalg.trace(
        tf.matmul(
            tf.cast(y_true, dtype='float32'),
            tf.transpose(tf.cast(tf.log(y_pred + 1e-10), dtype='float32'), [0, 1, 3, 2])
            ) +
        tf.matmul(
            tf.cast((1 - y_true), dtype='float32'),
            tf.transpose(tf.cast(tf.log(1 - y_pred + 1e-10), dtype='float32'), [0, 1, 3, 2])
            )
        )

def train_poly_music(ep, X_train, X_valid, X_test, train_losses, train_losses_no_reg, valid_losses, test_losses, size):
    epoch_losses = list()
    epoch_losses_no_reg = list()
    epoch_losses_valid = list()
    epoch_losses_test = list()
    
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]    
        x_train, y_train = (data_line[:-1]), (data_line[1:])
        
        loss = model.train_on_batch(
            x = x_train.reshape(1, 1,x_train.shape[0] , cf.n_classes),
            y = y_train.reshape(1, 1,x_train.shape[0] , cf.n_classes))
            #reset_metrics=False)
        epoch_losses.append(loss[0] / x_train.shape[0])
        epoch_losses_no_reg.append(loss[1] / x_train.shape[0])
        size.append(utils.effective_size(model))


    valid_idx_list = np.arange(len(X_valid), dtype="int32")
    for idx in valid_idx_list:
        data_line = X_valid[idx]
        
        x_valid, y_valid = (data_line[:-1]), (data_line[1:])
        valid_loss = model.test_on_batch(
            x = x_valid.reshape(1, 1,x_valid.shape[0] , cf.n_classes),
            y = y_valid.reshape(1, 1,x_valid.shape[0] , cf.n_classes))
        epoch_losses_valid.append(valid_loss[0] / x_valid.shape[0])
        
    #train_losses.append(sum(epoch_losses) / (1.0 * len(epoch_losses)))
    #train_losses_no_reg.append(sum(epoch_losses_no_reg) / (1.0 * len(epoch_losses_no_reg)))
    train_losses.append(epoch_losses)
    train_losses_no_reg.append(epoch_losses_no_reg)
 
    valid_losses.append(sum(epoch_losses_valid) / (1.0 * len(epoch_losses_valid)))
    print("Epoch : {e} \t Loss : {l} \t Val_Loss : {v}".format(e=ep, l=train_losses[ep], v=valid_losses[ep]))
        
    test_idx_list = np.arange(len(X_test), dtype="int32")
    for idx in test_idx_list:
        data_line = X_test[idx]
        
        x_test, y_test = (data_line[:-1]), (data_line[1:])
        test_loss = model.test_on_batch(
                x = x_test.reshape(1, 1,x_test.shape[0] , cf.n_classes),
                y = y_test.reshape(1, 1,x_test.shape[0] , cf.n_classes))
        epoch_losses_test.append(test_loss[0] / x_test.shape[0])
    
    test_losses.append(sum(epoch_losses_test) / (1.0 * len(epoch_losses_test)))
    print("Test loss : {}".format(test_losses[ep]))
    
    return train_losses[ep], train_losses_no_reg[ep], valid_losses[ep], test_losses[ep]

# training pipeline callbacks
if cf.dataset == 'PPG_Dalia':
    early_stop = EarlyStopping(monitor=val_mae, min_delta=0.01, patience=35, mode='min', verbose=1)
elif cf.dataset == 'copy_memory':
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=35, mode='min', verbose=1)
elif cf.dataset == 'poly_music':
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=35, mode='min', verbose=1)

warmup = WarmUp(patience=cf.warmup, strength=cf.reg_strength, l2=cf.l2)

save_gamma = SaveGamma()

strg_ctrl = strength_control(delta=5)

exp_str = export_structure()

log_dir = "./logs"

#tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#        histogram_freq=4, 
#        write_images=True,
#        write_grads=True,
#        update_freq='epoch') 

# obtain data from the choosen dataset
if cf.dataset == 'PPG_Dalia':
    X, y, groups, activity = pp.preprocessing(cf.dataset)
    '''
    X = np.random.randint(10, size=(64697,4,256))
    y = np.random.randint(10, size=(64697,1)) 
    groups = np.random.randint(10, size=(64697)) 
    activity = np.random.randint(10, size=(64697,1))
    '''
    
    # organize data
    group_kfold = GroupKFold(n_splits=4)
    group_kfold.get_n_splits(X, y, groups)
    
    predictions = dict()
    MAE = dict()
    dataset = dict()

    # Learn dil fact
    model = build_model.build_d1_custom(1, cf.input_shape, hyst=cf.hysteresis)
    del model
    model = build_model.build_d1_custom(1, cf.input_shape, hyst=cf.hysteresis, trainable=False)
    
    
    # save model and weights
    checkpoint = ModelCheckpoint(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5', monitor=val_mae, verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    #configure  model
    adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='logcosh', optimizer=adam, metrics=[mae])
    
    
    X_sh, y_sh = shuffle(X, y)

elif cf.dataset == 'copy_memory':
    n_train = 10000
    n_test = 1000
    x_train, y_train = data_generator(cf.T, cf.seq_len, n_train, cf.a)
    x_test, y_test = data_generator(cf.T, cf.seq_len, n_test, cf.a)
    x_valid = x_test
    y_valid = y_test
    
    # build model
    model = build_model_tcn.build_tcn_d1_custom(cf.input_shape, cf.n_classes, cf.n_channels, cf.k, cf.dp)
    del model
    model = build_model_tcn.build_tcn_d1_custom(cf.input_shape, cf.n_classes, cf.n_channels, cf.k, cf.dp,
                                                trainable=False)
    
    # save model and weights
    checkpoint = ModelCheckpoint(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

    #opt = Adam(lr=lr, clipvalue=1.0)
    opt = RMSprop(lr=cf.lr, clipvalue=1.0)
    
    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #loss=NLL,
            optimizer=opt) 
            #metrics=[mae])
elif cf.dataset == 'poly_music':
    #JSB_Chorales or Nottingham
    data = loadmat('./TCN/polymusic/Nottingham.mat')
    
    X_train = data['traindata'][0]
    X_valid = data['validdata'][0]
    X_test = data['testdata'][0]
    
    # build model
    model = build_model_tcn.build_tcn_d1_custom(1, cf.n_classes, 0, cf.n_channels, cf.k, cf.dp)
    del model
    model = build_model_tcn.build_tcn_d1_custom(1, cf.n_classes, 0, cf.n_channels, cf.k, cf.dp,
                                                trainable=False)
    
    opt = Adam(lr=cf.lr, clipvalue=0.4)
    #opt = RMSprop(lr=lr, clipvalue=1.0)
    
    model.compile(
            #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            loss=NLL,
            optimizer=opt) 
            #metrics=[mae])
 
# Training 
gamma_history = []

if cf.warmup != 0:
    #if not os.path.isfile(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5'):
    print('Train model for {} epochs'.format(cf.warmup))
    strg = cf.reg_strength
    cf.reg_strength = 0
    
    if cf.warmup == 'max':
        epochs_num = cf.epochs
    else:
        epochs_num = cf.warmup
    
    if cf.dataset == 'PPG_Dalia':
        hist = model.fit(x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1),                         (0, 3, 2, 1)), \
                             y=y_sh, shuffle=True, \
                             validation_split=0.1, verbose=1, \
                             batch_size= cf.batch_size, epochs=epochs_num, 
                             callbacks = [early_stop, checkpoint])
    elif cf.dataset == 'copy_memory':
        hist = model.fit(
            x = x_train.reshape(x_train.shape[0], 1, cf.n_steps, cf.input_shape),
            y = y_train,#.reshape(y_train.shape[0], 1, 1, n_steps), 
            epochs = epochs_num, 
            batch_size = cf.batch_size, 
            validation_data = (x_valid.reshape(x_valid.shape[0], 1, cf.n_steps, cf.input_shape), 
                               y_valid), 
            verbose=1, callbacks = [early_stop, checkpoint])
    elif cf.dataset == 'poly_music':
        train_losses = list()
        valid_losses = list()
        test_losses = list()
        
        best_v_l = np.Inf
        patience = 30
        wait = 0
        for ep in range(epochs_num):
            tr_l, v_l, t_l = train_poly_music(ep, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)
           
            if ep >= 10:
                if np.less(v_l, best_v_l):
                    print("Validation loss improved from {} to {}".format(best_v_l, v_l))
                    best_v_l = v_l
                    print("Save model to {}".format(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5'))
                    model.save_weights(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5')
                    wait = 0
                else:
                    wait += 1
                    print("Val loss did not improve from {}".format(best_v_l))
                    print("Iter for at least {} epochs".format(patience - wait))
                
                if wait >= patience:
                    print("Early Stop")
                    break
                
            
            ep += 1
        
    cf.reg_strength = strg
    #model.save_weights(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5')

del model
# build new model with trainable gamma
if cf.dataset == 'PPG_Dalia':
    model = build_model.build_d1_custom(1, cf.input_shape, trainable=True, hyst=cf.hysteresis)
    model.compile(loss='logcosh', optimizer=adam, metrics=[mae])
elif cf.dataset == 'copy_memory':
    model = build_model_tcn.build_tcn_d1_custom(cf.input_shape, cf.n_classes, cf.n_channels, cf.k, cf.dp,
                                                trainable=True)
    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #loss=NLL,
            optimizer=opt) 
            #metrics=[mae])
elif cf.dataset == 'poly_music':
    model = build_model_tcn.build_tcn_d1_custom(1, cf.n_classes, 0, cf.n_channels, cf.k, cf.dp,
                                                trainable=True)
    model.compile(
        loss=NLL,
        optimizer=opt,
        metrics=[NLL]) 



if cf.warmup != 0:
    # build temp model with non-trainable gamma
    if cf.dataset == 'PPG_Dalia':
        tmp_model = build_model.build_d1_custom(1, cf.input_shape, trainable=False, hyst=cf.hysteresis)
    elif cf.dataset == 'copy_memory':
        tmp_model = build_model_tcn.build_tcn_d1_custom(cf.input_shape, cf.n_classes, cf.n_channels, cf.k, cf.dp,
                                                trainable=False)
    elif cf.dataset == 'poly_music':
        tmp_model = build_model_tcn.build_tcn_d1_custom(1, cf.n_classes, 0, cf.n_channels, cf.k, cf.dp,
                                                trainable=False)
    
    # load weights in temp model
    tmp_model.load_weights(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5')

    # copy weights from tmp_model to model
    # this tedious step is necessary because keras save in last positions non-trainable
    # weights, thus passing from non-trainable to trainable generates a mismatch error
    # between shapes of array of weights
    weight_list = tmp_model.get_weights()
    for i, layer in enumerate(tmp_model.layers):
        if re.search('learned_conv2d.+_?[0-9]', layer.name):
            if cf.hysteresis:
                order = [2, 0, 1, 3]
            else:
                order = [2, 0, 1]
            ordered_w = [layer.get_weights()[i] for i in order]
            model.layers[i].set_weights(ordered_w)
        else:
            model.layers[i].set_weights(layer.get_weights())

model.save_weights(cf.saving_path+'autodil/test_trained_weights_warmup'+str(cf.warmup)+'.h5')

print('Train on Gammas')
print('Reg strength : {}'.format(cf.reg_strength))
if cf.dataset == 'PPG_Dalia':
    hist = model.fit(x=np.transpose(X_sh.reshape(X_sh.shape[0], 4, cf.input_shape, 1), (0, 3, 2, 1)), \
                         y=y_sh, shuffle=True, \
                         validation_split=0.1, verbose=1, \
                         batch_size= cf.batch_size, epochs=cf.epochs, 
                         callbacks = [early_stop, save_gamma, exp_str])
elif cf.dataset == 'copy_memory':
    hist = model.fit(
            x = x_train.reshape(x_train.shape[0], 1, cf.n_steps, cf.input_shape),
            y = y_train,#.reshape(y_train.shape[0], 1, 1, n_steps), 
            epochs = cf.epochs, 
            batch_size = cf.batch_size, 
            validation_data = (x_valid.reshape(x_valid.shape[0], 1, cf.n_steps, cf.input_shape), 
                               y_valid), 
            verbose=1,  callbacks = [early_stop, save_gamma, exp_str])
elif cf.dataset == 'poly_music':
    train_losses = list()
    train_losses_no_reg = list()
    valid_losses = list()
    test_losses = list()
   
    total_loss = list()
    total_loss_no_reg = list()
    reg_loss = list()
    size = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0
    gamma = dict()

    # add loss without reg to model's metrics
    #loss_no_reg = model.total_loss - sum(model.losses)
    #model.metrics_tensor.append(loss_no_reg)
    #model.metrics_names.append('loss_no_reg')

    size.append(model.count_params())
    
    for ep in range(cf.epochs):
        tr_l, tr_nr_l, v_l, t_l = train_poly_music(ep, X_train, X_valid, X_test, train_losses, train_losses_no_reg, valid_losses, test_losses, size)
        
        total_loss.append(tr_l)
        total_loss_no_reg.append(tr_nr_l)
        reg_loss.append([l1-l2 for (l1,l2) in zip(tr_l, tr_nr_l)])
        #pdb.set_trace()
        
        #size.append(utils.effective_size(model))

        if ep >= 10:
            # early_stop + export_structure
            if np.less(v_l, best_v_l):
                if v_l >= 2:
                    best_v_l = v_l
                #model.save_weights(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5')
                wait = 0
                
                # Record the best model if current results is better (less).
                names = [weight.name for layer in model.layers for weight in layer.weights]
                weights = model.get_weights()
                for name, weight in zip(names, weights):
                    if re.search('learned_conv2d.+_?[0-9]/gamma', name):
                        gamma[name] = weight
                        gamma[name] = np.array(gamma[name] > cf.threshold, dtype=bool)
                        gamma[name] = utils.dil_fact(gamma[name], op='mul')
                
                print("New best MAE, update file. \n")    
                print(gamma) 
                utils.save_dil_fact(cf.saving_path, gamma)
                
            else:
                wait += 1
                print("Val loss did not improve from {}".format(best_v_l))
                print("Iter for at least {} epochs".format(patience - wait))
            
        if wait >= patience:
            print("Early Stop")
            break
        
        ep += 1
    
model.save_weights(cf.saving_path+'autodil/test2_trained_weights_warmup'+str(cf.warmup)+'.h5')


#loss_size_dict = dict()
#
#loss_size_dict['Loss'] = total_loss
#loss_size_dict['Loss_n_r'] = total_loss_no_reg
#loss_size_dict['Reg_loss'] = reg_loss
#loss_size_dict['Size'] = size

# save predictions and real values
#with open('./size_loss_poly_music'+'_'+str(cf.reg_strength)+'_'+str(cf.threshold)+'.pkl', 'wb') as handle:
#    pickle.dump(loss_size_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

##plot history for loss
#plt.figure()
#plt.plot(total_loss[5:], label='loss')
#plt.plot(total_loss_no_reg[5:], label='loss_no_reg')
#plt.plot(reg_loss[5:], label='reg_loss')
#plt.title('Model Loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(loc='upper left')
#plt.savefig('loss_{}_{}_{}.png'.format(cf.reg_strength, cf.threshold, cf.warmup), format='png', dpi=800)
#plt.show()
#
## plot size trend over training
#plt.figure()
#plt.plot(size, label='size')
#plt.title('Model Size')
#plt.ylabel('Size')
#plt.xlabel('Epoch')
##plt.legend(loc='upper left')
#plt.savefig('size_{}_{}_{}.png'.format(cf.reg_strength, cf.threshold, cf.warmup), format='png', dpi=800)
#plt.show()
#
#
#
## plot history for mae
#plt.figure()
#plt.plot(hist.history['mean_absolute_error'][5:], label='mae')
#plt.plot(hist.history['val_mean_absolute_error'][5:], label='val_mae')
#plt.title('model mae')
#plt.ylabel('mae')
#plt.xlabel('epoch')
#plt.legend(loc='upper left')
#plt.savefig('mae_{}_{}_{}.png'.format(cf.reg_strength, cf.threshold, cf.warmup), format='png', dpi=1200)
#plt.show()
#
## plot history for gamma
#gamma_traces = dict()
#
#for i in range(7):
#    gamma_traces[i] = []
#
#    for diz in gamma_history:
#        gamma_traces[i].append(diz[i])
#
#    plt.figure()
#    plt.xlabel("epoch")
#    plt.ylabel("gamma")
#    plt.title('layer'+str(i))
#    plt.plot(gamma_traces[i], '^-')
#    plt.legend(['gamma'+str(j) for j in range(i//2+2)])
#    plt.savefig('gamma'+str(i)+'_{}_{}_{}.png'.format(cf.reg_strength, cf.threshold, cf.warmup), format='png', dpi=800)
#    plt.show()

if cf.dataset == 'PPG_Dalia':    
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
                #dil_list = dil_list[2:] + dil_list[:2]
    
            model = build_model.build_dn_learned_dil(1, cf.input_shape, dil_list)
            #model = build_model.build_dn_learned_dil(1, cf.input_shape)
            #model.add_loss(lambda: tf.losses.get_regularization_loss())
    
            # save model and weights
            checkpoint = ModelCheckpoint(cf.saving_path+'test_reg'+str(k)+'.h5', monitor=val_mae, verbose=1,\
                             save_best_only=True, save_weights_only=False, mode='min', period=1)
            #configure  model
            adam = Adam(lr=cf.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
            #model.compile(loss=mean_squared_error, optimizer=adam, metrics=['mae'])
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
            
            dataset['P'+str(k)+'_label'] = y_test
            dataset['P'+str(k)+'_pred'] = predictions[k]
            dataset['P'+str(k)+'_activity'] = activity_test
            
            
            print(MAE)  
    
    # save predictions and real values
    with open(cf.saving_path+'dataset/'+'test_reg'+'_'+str(cf.reg_strength)+'_'+str(cf.threshold)+'.pkl', 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(MAE)
    # Evaluate average MAE
    avg = 0
    for _, val in MAE.items():
        avg += val
    print("Average MAE : %f", avg/len(MAE))
    
    # summary file
    f=open("summary_warmup{}.txt".format(cf.warmup), "a+")
    
    f.write("regularization strength : {reg_str} \t threshold : {th} \t MAE : {mae} \t Model size : {size} \n".format(
        reg_str = cf.reg_strength,
            th = cf.threshold,
            mae = avg/len(MAE),
            size = model.count_params()))
    
    f.close()

elif cf.dataset == 'copy_memory':
    # obtain conv #output filters from learned json structure
    with open(cf.saving_path+'autodil/learned_dil_'+'{:.1e}'.format(cf.reg_strength)+'_'+'{:.1e}'.format(cf.threshold)+'_{}'.format(cf.warmup)+'.json', 'r') as f:
        dil_list = [val for _,val in json.loads(f.read()).items()]
      
    del model
    model = build_model_tcn.build_tcn_learned_dil(cf.input_shape, cf.n_classes, cf.n_channels, cf.k, cf.dp, dil_list)
    #model = build_model_tcn.build_tcn_learned_dil(1, cf.input_shape, dil_list)
    model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            #loss=NLL,
            optimizer=opt) 
    
    hist = model.fit(
            x = x_train.reshape(x_train.shape[0], 1, cf.n_steps, cf.input_shape),
            y = y_train,#.reshape(y_train.shape[0], 1, 1, n_steps), 
            epochs = cf.epochs, 
            batch_size = cf.batch_size, 
            validation_data = (x_valid.reshape(x_valid.shape[0], 1, cf.n_steps, cf.input_shape), 
                               y_valid), 
            verbose=1, callbacks=[checkpoint])
    
    #evaluate
    results = model.evaluate(x = x_test.reshape(x_test.shape[0], 1, cf.n_steps, cf.input_shape), 
                        y = y_test,#.reshape(y_test.shape[0], 1, 1, n_steps), 
                        batch_size = cf.batch_size)
    
    # summary file
    f=open("summary_copy_mem_warmup{}.txt".format(cf.warmup), "a+")
    
    f.write("regularization strength : {reg_str} \t threshold : {th} \t Loss : {loss} \t Model size : {size} \n".format(
        reg_str = cf.reg_strength,
            th = cf.threshold,
            loss = min(hist.history['val_loss']),
            size = model.count_params()))
    
    f.close()
elif cf.dataset == 'poly_music':
    # obtain conv #output filters from learned json structure
    with open(cf.saving_path+'autodil/learned_dil_'+'{:.1e}'.format(cf.reg_strength)+'_'+'{:.1e}'.format(cf.threshold)+'_{}'.format(cf.warmup)+'.json', 'r') as f:
        dil_list = [val for _,val in json.loads(f.read()).items()]
      
    del model
    model = build_model_tcn.build_tcn_learned_dil(1, cf.n_classes, 0, cf.n_channels, cf.k,cf.dp,dil_list=dil_list)
    
    model.compile(
        loss=NLL,
        optimizer=opt) 
    
    train_losses = list()
    valid_losses = list()
    test_losses = list()
    
    best_v_l = np.Inf
    patience = 30
    wait = 0
    for ep in range(cf.epochs):
        tr_l, v_l, t_l = train_poly_music(ep, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)
        
        if ep >= 10:
            if np.less(v_l, best_v_l):
                best_v_l = v_l
                model.save_weights(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5')
                wait = 0
                best_t_l = t_l
            else:
                wait += 1
                print("Val loss did not improve from {}".format(best_v_l))
                print("Iter for at least {} epochs".format(patience - wait))
            
            if wait >= patience:
                print("Early Stop")
                break
            
            
            ep += 1
    
    # summary file
    f=open("summary_poly_music_warmup{}.txt".format(cf.warmup), "a+")
    
    f.write("regularization strength : {reg_str} \t threshold : {th} \t Loss : {loss} \t Model size : {size} \n".format(
        reg_str = cf.reg_strength,
            th = cf.threshold,
            loss = best_t_l,
            size = model.count_params()))
    
    f.close()
