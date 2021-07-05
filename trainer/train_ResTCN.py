# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:36:50 2020

@author: MatteoRisso
"""

import numpy as np
#import config as cf
import re
import utils
import math
import tensorflow as tf
import pdb

def train_Nottingham(model, ep, cf, X_train, X_valid, X_test, train_losses, valid_losses, test_losses):
    epoch_losses = list()
    epoch_losses_valid = list()
    epoch_losses_test = list()

    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx].astype('float32')
        x_train, y_train = (data_line[:-1]), (data_line[1:])

        loss = model.train_on_batch(
           x = x_train.reshape(1, 1, x_train.shape[0], cf.n_classes),
           y = y_train.reshape(1, 1, x_train.shape[0], cf.n_classes))
        
        if math.isnan(loss):
            pdb.set_trace()
            #tf.keras.models.save_model(model, 'model_'+str('nan')+'.h5')
            #break
            #raise ValueError
        else:
            pass
            #tf.keras.models.save_model(model, 'model_'+str('pre_nan')+'.h5')
            
        epoch_losses.append(loss / x_train.shape[0])

    valid_idx_list = np.arange(len(X_valid), dtype="int32")
    for idx in valid_idx_list:
        data_line = X_valid[idx].astype('float32')

        x_valid, y_valid = (data_line[:-1]), (data_line[1:])
        valid_loss = model.test_on_batch(
            x = x_valid.reshape(1, 1, x_valid.shape[0], cf.n_classes),
            y = y_valid.reshape(1, 1, x_valid.shape[0], cf.n_classes))
        epoch_losses_valid.append(valid_loss / x_valid.shape[0])

    train_losses.append(sum(epoch_losses) / (1.0 * len(epoch_losses)))
    valid_losses.append(sum(epoch_losses_valid) / (1.0 * len(epoch_losses_valid)))
    print("Epoch : {e} \t Loss : {l} \t Val_Loss : {v}".format(e=ep, l=train_losses[ep], v=valid_losses[ep]))
    
    if math.isnan(valid_losses[ep]):
        pdb.set_trace()
    
    test_idx_list = np.arange(len(X_test), dtype="int32")
    for idx in test_idx_list:
        data_line = X_test[idx].astype('float32')

        x_test, y_test = (data_line[:-1]), (data_line[1:])
        test_loss = model.test_on_batch(
            x = x_test.reshape(1, 1,x_test.shape[0] , cf.n_classes),
            y = y_test.reshape(1, 1,x_test.shape[0] , cf.n_classes))
        epoch_losses_test.append(test_loss / x_test.shape[0])

    test_losses.append(sum(epoch_losses_test) / (1.0 * len(epoch_losses_test)))
    print("Test loss : {}".format(test_losses[ep]))

    return train_losses[ep], valid_losses[ep], test_losses[ep]

def warmup_Nottingham(model, epochs_num, cf, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Nottingham(model, ep, cf, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)

        if ep >= 2:
            if np.less(v_l, best_v_l):
                print("Validation loss improved from {} to {}".format(best_v_l, v_l))
                best_v_l = v_l
                print("Save model to {}".format(cf.saving_path+cf.dataset+'/trained_weights_warmup'+str(cf.warmup)+'.h5'))
                model.save_weights(cf.saving_path+cf.dataset+'/trained_weights_warmup'+str(cf.warmup)+'.h5')
                wait = 0
            else:
                wait += 1
                print("Val loss did not improve from {}".format(best_v_l))
                print("Iter for at least {} epochs".format(patience - wait))

            if wait >= patience:
                print("Early Stop")
                break
        ep += 1

    return

def warmup_SeqMNIST(model, epochs_num, cf, X_train, y_train, early_stop, checkpoint):
    hist = model.fit(
            x=X_train.reshape(-1, 1, X_train.shape[-1], 1),
            y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
            validation_split=0.1, verbose=1, 
            batch_size= cf.batch_size, epochs=epochs_num,
            callbacks = [early_stop, checkpoint])
    return

def train_gammas_Nottingham(model, epochs, cf, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0
    gamma = dict()

    for ep in range(epochs):
        tr_l, v_l, t_l = train_Nottingham(model, ep, cf, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)

        if ep >= 10:
            # early_stop + export_structure
            if np.less(v_l, best_v_l):
                if v_l >= 2:
                    best_v_l = v_l
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
                utils.save_dil_fact(cf.saving_path+cf.dataset, gamma, cf)

            else:
                wait += 1
                print("Val loss did not improve from {}".format(best_v_l))
                print("Iter for at least {} epochs".format(patience - wait))

        if wait >= patience:
            print("Early Stop")
            break

        ep += 1

def train_gammas_SeqMNIST(model, cf, X_train, y_train, early_stop, save_gamma, exp_str): 
    hist = model.fit(
            x=X_train.reshape(-1, 1, X_train.shape[-1], 1),
            y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
            validation_split=0.1, verbose=1, 
            batch_size= cf.batch_size, epochs=cf.epochs,
            callbacks = [early_stop, save_gamma, exp_str])
    return

def retrain_Nottingham(model, epochs_num, cf, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Nottingham(model, ep, cf, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)

        if ep >= 10:
            if np.less(v_l, best_v_l):
                print("Validation loss improved from {} to {}".format(best_v_l, v_l))
                best_v_l = v_l
                print("Save model to {}".format(cf.saving_path+cf.dataset+'/trained_weights_warmup'+str(cf.warmup)+'.h5'))
                model.save_weights(cf.saving_path+cf.dataset+'/trained_weights_warmup'+str(cf.warmup)+'.h5')
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

    return best_t_l

def retrain_SeqMNIST(model, epochs_num, cf, X_train, y_train, X_test, y_test, early_stop, checkpoint):
    hist = model.fit(
        x=X_train.reshape(-1, 1, X_train.shape[-1], 1),
        y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
        validation_data=(X_test.reshape(-1, 1, X_test.shape[-1], 1), y_test.reshape(y_test.shape[0], 1)), 
        verbose=1, 
        batch_size= cf.batch_size, epochs=cf.epochs,
        callbacks = [early_stop, checkpoint])
    return


