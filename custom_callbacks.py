# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:11:50 2020

@author: MatteoRisso
"""

import config as cf
import numpy as np
import tensorflow as tf
import utils

import re
import sys

if tf.__version__ == '1.14.0':
    # some aliases necessary in tf 1.14
    val_mae = 'val_mean_absolute_error'
    mae = 'mean_absolute_error'
else:
    # some aliases necessary in tf > 1.14
    val_mae = 'val_mean_absolute_error'
    mae = 'mean_absolute_error'

class export_structure(tf.keras.callbacks.Callback):
    def __init__(self):
        super(export_structure, self).__init__()
    
    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        if cf.dataset == 'PPG_Dalia':
           self.best = np.Inf
        elif cf.dataset == 'Nottingham' or cf.dataset == 'JSB_Chorales':
           self.best = np.Inf    
        elif cf.dataset == 'SeqMNIST' or cf.dataset == 'PerMNIST':
           self.best = 0
        else:
           print("{} is not supported".format(cf.dataset))
           sys.exit()
        self.gamma = dict()
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        #get current validation mae
        if cf.dataset == 'PPG_Dalia':
            current = logs.get(val_mae)
            l = 1
            h = 0
            wait = 0
        elif cf.dataset == 'Nottingham' or cf.dataset == 'JSB_Chorales':
            current = logs.get("val_loss")
            l = 1
            h = 0
        elif cf.dataset == 'SeqMNIST' or cf.dataset == 'PerMNIST':
            current = logs.get("val_accuracy")
            l = 0
            h = 1
            wait = 20
        else:
            print("{} is not supported".format(cf.dataset))
            sys.exit()
	
        if self.i > wait:
            # compare with previous best one
            if bool(np.less(current, self.best) * l) ^ \
                bool((current > self.best) * h):
                self.best = current

                # Record the best model if current results is better.
                names = [weight.name for layer in self.model.layers for weight in layer.weights]
                weights = self.model.get_weights()
                for name, weight in zip(names, weights):
                    if re.search('learned_conv2d.+_?[0-9]/gamma', name):
                        self.gamma[name] = weight
                        self.gamma[name] = np.array(self.gamma[name] > cf.threshold, dtype=bool)
                        self.gamma[name] = utils.dil_fact(self.gamma[name], op='mul')
                    elif re.search('weight_norm.+_?[0-9]/gamma', name):
                        self.gamma[name] = weight
                        self.gamma[name] = np.array(self.gamma[name] > cf.threshold, dtype=bool)
                        self.gamma[name] = utils.dil_fact(self.gamma[name], op='mul')
                print("New best model, update file. \n")
                print(self.gamma)
                utils.save_dil_fact(cf.saving_path, self.gamma)
        else:
            self.i += 1

class SaveGamma(tf.keras.callbacks.Callback):
    
    def set_model(self, model):
        self.model = model

    def on_epoch_end(self, epoch, logs):

        names = [weight.name for layer in self.model.layers for weight in layer.weights]
        weights = self.model.get_weights()

        gamma = dict()
        i = 0
        for name, weight in zip(names, weights):
            if re.search('learned_conv2d.+_?[0-9]/gamma', name):
                print('gamma: ', weight.tolist()[0])
                gamma[i] = weight.tolist()[0]
                i += 1
            elif re.search('weight_norm.+_?[0-9]/gamma', name):
                print('gamma: ', weight.tolist()[0])
                gamma[i] = weight.tolist()[0]
                i += 1
        #gamma_history.append(gamma)
