#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Matteo Risso                                                      *
#*----------------------------------------------------------------------------*

#import config as cf
import numpy as np
import tensorflow as tf
import utils

import re
import sys

# aliases
val_mae = 'val_mean_absolute_error'
mae = 'mean_absolute_error'

class export_structure(tf.keras.callbacks.Callback):
    def __init__(self, cf):
        self.cf = cf
        super(export_structure, self).__init__()
    
    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        if self.cf.dataset == 'PPG_Dalia':
           self.best = np.Inf
        elif self.cf.dataset == 'Nottingham' or self.cf.dataset == 'JSB_Chorales':
           self.best = np.Inf    
        elif self.cf.dataset == 'SeqMNIST' or self.cf.dataset == 'PerMNIST':
           self.best = 0
        else:
           print("{} is not supported".format(self.cf.dataset))
           sys.exit()
        self.gamma = dict()
        self.i = 0

    def on_epoch_end(self, epoch, logs=None):
        #get current validation mae
        if self.cf.dataset == 'PPG_Dalia':
            current = logs.get(val_mae)
            l = 1
            h = 0
            wait = 0
        elif self.cf.dataset == 'Nottingham' or self.cf.dataset == 'JSB_Chorales':
            current = logs.get("val_loss")
            l = 1
            h = 0
        elif self.cf.dataset == 'SeqMNIST' or self.cf.dataset == 'PerMNIST':
            current = logs.get("val_accuracy")
            l = 0
            h = 1
            wait = 20
        else:
            print("{} is not supported".format(self.cf.dataset))
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
                        self.gamma[name] = np.array(self.gamma[name] > self.cf.threshold, dtype=bool)
                        self.gamma[name] = utils.dil_fact(self.gamma[name], op='mul')
                    elif re.search('weight_norm.+_?[0-9]/gamma', name):
                        self.gamma[name] = weight
                        self.gamma[name] = np.array(self.gamma[name] > self.cf.threshold, dtype=bool)
                        self.gamma[name] = utils.dil_fact(self.gamma[name], op='mul')
                print("New best model, update file. \n")
                print(self.gamma)
                if self.cf.dataset == 'PPG_Dalia':
                    utils.save_dil_fact(self.cf.saving_path+self.cf.dataset, self.gamma, self.cf)
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
