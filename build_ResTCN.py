# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 18:43:14 2020

@author: MatteoRisso
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from math import ceil

import tensorflow.keras.backend as K

from tensorflow_probability.python.layers.weight_norm import WeightNorm

from auto_layers import LearnedConv2D, WeightNormConv2D, DenseTied

def log_softmax(x):
    return tf.nn.log_softmax(x)

def ResTCN(n_classes, n_channels, k=6, dp=0.2, variant='Nottingham'):
    
    if variant == 'Nottingham' or variant == 'JSB_Chorales':
        inputs = layers.Input(shape=(1, None, n_classes))
    elif variant == 'SeqMNIST' or variant == 'PerMNIST':
        inputs = layers.Input(shape=(1, 784, 1))
    elif variant == 'AddProb':
        inputs = layers.Input(shape=(1, 600, n_classes))
    elif variant == 'Word_PTB':
        n_words = n_classes[0]
        emb_size = n_classes[1]

        
        real_inputs = layers.Input(shape=(1, None))
        emb_layer = layers.Embedding(n_words, emb_size,
                               embeddings_initializer=RandomNormal(mean=0.0, stddev=0.01),
                               input_length=80) 
        emb = emb_layer(real_inputs)
        inputs = layers.Dropout(0.25) (emb)
    elif variant == 'Char_PTB':
        n_words = n_classes[0]
        emb_size = n_classes[1]
        
        real_inputs = layers.Input(shape=(1, None))
        emb_layer = layers.Embedding(n_words, emb_size,
                               embeddings_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                               input_length=400) 
        emb = emb_layer(real_inputs)
        inputs = layers.Dropout(0.1) (emb)
        
    n_levels = len(n_channels)
    
    for i in range(n_levels):
        dilation_size = 2 ** i
        padding = (k - 1) * dilation_size
        in_channels = n_classes if i == 0 else n_channels[i-1]
        out_channels = n_channels[i]
        
        # TemporalBlock
        if i == 0:
            x = inputs
            y = layers.ZeroPadding2D(padding=((0, 0), (padding, 0))) (x)
            y = WeightNorm(layers.Conv2D(
                    filters=out_channels, 
                    kernel_size=(1,k), 
                    padding='valid',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01), 
                    dilation_rate=(1, dilation_size))) (y)
        else:
            y = layers.ZeroPadding2D(padding=((0, 0), (padding, 0))) (x)
            y = WeightNorm(layers.Conv2D(
                    filters=out_channels, 
                    kernel_size=(1,k), 
                    padding='valid',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    dilation_rate=(1, dilation_size))) (y)
        
        y = layers.ReLU() (y)
        y = layers.Dropout(dp) (y)
        
        y = layers.ZeroPadding2D(padding=((0, 0), (padding, 0))) (y)
        y = WeightNorm(layers.Conv2D(
                filters=out_channels, 
                kernel_size=(1,k), 
                padding='valid',
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                dilation_rate=(1, dilation_size))) (y)
        
        y = layers.ReLU() (y)
        y = layers.Dropout(dp) (y)
        
        if in_channels != out_channels:
            x = layers.Conv2D(
                    filters=out_channels, 
                    kernel_size=(1,1),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    padding='valid') (x)
    
        x = layers.Add() ([x, y])
        x = layers.ReLU() (x)
    
    if variant == 'Nottingham' or variant == 'JSB_Chorales':
        x = layers.Dense(n_classes, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         input_shape=(1,None,out_channels)) (x)
        x = layers.Activation('sigmoid') (tf.cast(x,dtype='float64'))  
        model = Model(inputs, x)
    elif variant == 'SeqMNIST' or variant == 'PerMNIST':
        x = layers.Lambda(lambda x : x[:,:,-1,:]) (x)
        x = layers.Dense(n_classes, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         input_shape=(1,784,out_channels)) (x)
        x = layers.Softmax() (x)
        model = Model(inputs, x)
    elif variant == 'AddProb':
        x = layers.Dense(n_classes-1, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)) (x)
        x = layers.Lambda(lambda x : x[:,:,-1,:]) (x)
        model = Model(inputs, x)
    elif variant == 'Word_PTB':
        # x = layers.Dense(n_words, 
        #                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)) (x)
        x = DenseTied(n_words, 
                      #kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                      tied_to = emb_layer) (x)
        #x = layers.Lambda(lambda x : x[:,:,40:,:]) (x)
        model = Model(real_inputs, x)
    elif variant == 'Char_PTB':
        # x = layers.Dense(n_words, 
        #                  kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
        #                  bias_initializer=RandomUniform(minval=0., maxval=0.)) (x)
        x = DenseTied(n_words, 
                      #kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                      tied_to = emb_layer) (x)
        x = layers.Lambda(lambda x : x[:,:,80:,:]) (x)
        model = Model(real_inputs, x)
    
    model.summary()
        
    return model

def ResTCN_d1(n_classes, n_channels, k=6, dp=0.2, trainable=True, variant='Nottingham', hyst=0):
    
    if variant == 'Nottingham' or variant == 'JSB_Chorales':
        inputs = layers.Input(shape=(1, None, n_classes))
    elif variant == 'SeqMNIST' or variant == 'PerMNIST':
        inputs = layers.Input(shape=(1, None, 1))
    elif variant == 'AddProb':
        inputs = layers.Input(shape=(1, 600, n_classes))
    elif variant == 'Word_PTB':
        n_words = n_classes[0]
        emb_size = n_classes[1]
        
        real_inputs = layers.Input(shape=(1,None))
        emb = layers.Embedding(n_words, emb_size) (real_inputs)
        inputs = layers.Dropout(0.25) (emb)
    elif variant == 'Char_PTB':
        n_words = n_classes[0]
        emb_size = n_classes[1]
        
        real_inputs = layers.Input(shape=(1, None))
        emb_layer = layers.Embedding(n_words, emb_size,
                               embeddings_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                               input_length=400) 
        emb = emb_layer(real_inputs)
        inputs = layers.Dropout(0.1) (emb)
    
    num_levels = len(n_channels)
    
    for i in range(num_levels):
        dilation_size = 2 ** i
        padding = (k - 1) * dilation_size
        in_channels = n_classes if i == 0 else n_channels[i-1]
        out_channels = n_channels[i]
        
        # TemporalBlock
        if i == 0:
            x = inputs
            y = layers.ZeroPadding2D(padding=((0, 0), (padding, 0))) (x)
            #y = WeightNorm(LearnedConv2D(
            y = LearnedConv2D(
                    filters=out_channels,
                    gamma_trainable=trainable, 
                    kernel_size=(1,(k-1)*dilation_size+1), 
                    padding='valid',
                    hyst = hyst,
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    dilation_rate=(1, 1)) (y)
        else:
            y = layers.ZeroPadding2D(padding=((0, 0), (padding, 0))) (x)
            #y = WeightNorm(LearnedConv2D(
            y = LearnedConv2D(
                    filters=out_channels,
                    gamma_trainable=trainable,
                    kernel_size=(1,(k-1)*dilation_size+1), 
                    padding='valid',
                    hyst=hyst,
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    dilation_rate=(1, 1)) (y)
            
        y = layers.ReLU() (y)
        y = layers.Dropout(dp) (y)
        
        y = layers.ZeroPadding2D(padding=((0, 0), (padding, 0))) (y)
        #y = WeightNorm(LearnedConv2D(
        y = LearnedConv2D(
                filters=out_channels,
                gamma_trainable=trainable,
                kernel_size=(1,(k-1)*dilation_size+1), 
                padding='valid',
                hyst=hyst,
                kernel_regularizer=tf.keras.regularizers.l2(),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                dilation_rate=(1, 1)) (y)
            
        y = layers.ReLU() (y)
        y = layers.Dropout(dp) (y)
        
        if in_channels != out_channels:
            x = layers.Conv2D(
                    filters=out_channels, 
                    kernel_size=(1,1),
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    padding='valid') (x)
    
        x = layers.Add() ([x, y])
        x = layers.ReLU() (x)
        
    if variant == 'Nottingham' or variant == 'JSB_Chorales':
        x = layers.Dense(n_classes, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         input_shape=(1,None,out_channels)) (x)
        x = layers.Activation('sigmoid') (tf.cast(x,dtype='float64'))  
        model = Model(inputs, x)
    elif variant == 'SeqMNIST' or variant == 'PerMNIST':
        x = layers.Lambda(lambda x : x[:,:,-1,:]) (x)
        x = layers.Dense(n_classes, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         input_shape=(1,None,out_channels)) (x)
        #x = layers.Lambda(log_softmax) (x)    
        x = layers.Softmax() (x) 
        model = Model(inputs, x)
    elif variant == 'AddProb':
        x = layers.Dense(n_classes-1, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)) (x)
        x = layers.Lambda(lambda x : x[:,:,-1,:]) (x)
        model = Model(inputs, x)
    elif variant == 'Word_PTB':
        x = layers.Dense(n_words, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)) (x)
        #x = layers.Lambda(lambda x : x[:,:,40:,:]) (x)
        model = Model(real_inputs, x)
    elif variant == 'Char_PTB':
        # x = layers.Dense(n_words, 
        #                  kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
        #                  bias_initializer=RandomUniform(minval=0., maxval=0.)) (x)
        x = DenseTied(n_words, 
                      #kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                      tied_to = emb_layer) (x)
        x = layers.Lambda(lambda x : x[:,:,80:,:]) (x)
        model = Model(real_inputs, x)
       
    model.summary()
        
    return model

def ResTCN_learned(n_classes, n_channels, k=6, dp=0.2, dil_list=[], variant='Nottingham'):
    
    if variant == 'Nottingham' or variant == 'JSB_Chorales':
        inputs = layers.Input(shape=(1, None, n_classes))
    elif variant == 'SeqMNIST' or variant == 'PerMNIST':
        inputs = layers.Input(shape=(1, None, 1))
    elif variant == 'AddProb':
        inputs = layers.Input(shape=(1, 600, n_classes))
    elif variant == 'Word_PTB':
        n_words = n_classes[0]
        emb_size = n_classes[1]
        
        real_inputs = layers.Input(shape=(1,None))
        emb = layers.Embedding(n_words, emb_size) (real_inputs)
        inputs = layers.Dropout(0.25) (emb)
    elif variant == 'Char_PTB':
        n_words = n_classes[0]
        emb_size = n_classes[1]
        
        real_inputs = layers.Input(shape=(1, None))
        emb_layer = layers.Embedding(n_words, emb_size,
                               embeddings_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                               input_length=400) 
        emb = emb_layer(real_inputs)
        inputs = layers.Dropout(0.1) (emb)
    
    num_levels = len(n_channels)
    cnt = 0
    
    for i in range(num_levels):
        dilation_size = 2 ** i
        padding1 = (ceil(((k-1)*dilation_size+1)/dil_list[cnt]) - 1) * dil_list[cnt]
        padding2 = (ceil(((k-1)*dilation_size+1)/dil_list[cnt+1]) - 1) * dil_list[cnt+1]
        in_channels = n_classes if i == 0 else n_channels[i-1]
        out_channels = n_channels[i]
        
        # TemporalBlock
        if i == 0:
            x = inputs
            y = layers.ZeroPadding2D(padding=((0, 0), (padding1, 0))) (x)
            y = WeightNorm(layers.Conv2D(
                    filters=out_channels, 
                    kernel_size=(1,ceil(((k-1)*dilation_size+1)/dil_list[cnt])), 
                    padding='valid',
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    dilation_rate=(1, int(dil_list[cnt])))) (y)
    
            cnt += 1
        else:
            y = layers.ZeroPadding2D(padding=((0, 0), (padding1, 0))) (x)
            y = WeightNorm(layers.Conv2D(
                    filters=out_channels, 
                    kernel_size=(1,ceil(((k-1)*dilation_size+1)/dil_list[cnt])),
                    padding='valid',
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    dilation_rate=(1, int(dil_list[cnt])))) (y)
    
            cnt += 1
            
        y = layers.ReLU() (y)
        y = layers.Dropout(dp) (y)
        
        y = layers.ZeroPadding2D(padding=((0, 0), (padding2, 0))) (y)
        y = WeightNorm(layers.Conv2D(
                filters=out_channels, 
                kernel_size=(1,ceil(((k-1)*dilation_size+1)/dil_list[cnt])),  
                padding='valid',
                kernel_regularizer=tf.keras.regularizers.l2(),
                kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                dilation_rate=(1, int(dil_list[cnt])))) (y)
        cnt += 1
            
        y = layers.ReLU() (y)
        y = layers.Dropout(dp) (y)
        
        if in_channels != out_channels:
            x = layers.Conv2D(
                    filters=out_channels, 
                    kernel_size=(1,1),
                    kernel_regularizer=tf.keras.regularizers.l2(),
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                    padding='valid') (x)

        x = layers.Add() ([x, y])
        x = layers.ReLU() (x)
        
    if variant == 'Nottingham' or variant == 'JSB_Chorales':
        x = layers.Dense(n_classes, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         input_shape=(1,None,out_channels)) (x)
        x = layers.Activation('sigmoid') (tf.cast(x,dtype='float64'))  
        model = Model(inputs, x)
    elif variant == 'SeqMNIST' or variant == 'PerMNIST':
        x = layers.Lambda(lambda x : x[:,:,-1,:]) (x)
        x = layers.Dense(n_classes, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
                         input_shape=(1,None,out_channels)) (x)
        #x = layers.Lambda(log_softmax) (x)
        x = layers.Softmax() (x) 
        model = Model(inputs, x)   
    elif variant == 'AddProb':
        x = layers.Dense(n_classes-1, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)) (x)
        x = layers.Lambda(lambda x : x[:,:,-1,:]) (x)
        model = Model(inputs, x)
    elif variant == 'Word_PTB':
        x = layers.Dense(n_words, 
                         kernel_initializer=RandomNormal(mean=0.0, stddev=0.01)) (x)
        #x = layers.Lambda(lambda x : x[:,:,40:,:]) (x)
        model = Model(real_inputs, x)
    elif variant == 'Char_PTB':
        # x = layers.Dense(n_words, 
        #                  kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
        #                  bias_initializer=RandomUniform(minval=0., maxval=0.)) (x)
        x = DenseTied(n_words, 
                      #kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1),
                      tied_to = emb_layer) (x)
        x = layers.Lambda(lambda x : x[:,:,80:,:]) (x)
        model = Model(real_inputs, x)
    
    
    model.summary()
        
    return model
