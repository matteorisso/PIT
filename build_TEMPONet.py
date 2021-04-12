# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:12:33 2020

@author: MatteoRisso
"""

from tensorflow.keras import Sequential, layers
from auto_layers import LearnedConv2D
import config as cf
import math

def TEMPONet_d1(width_mult, in_shape, trainable=True, ofmap=[]):
    
    input_channel = width_mult * 32
    output_channel = input_channel * 2
    
    if not ofmap:
        ofmap = [
                32, 32, 62,
                63, 64, 128,
                91, 35, 43,
                64, 56, 1
                ]

    model = Sequential()
    model.add(LearnedConv2D(
        gamma_trainable=trainable, 
        filters=ofmap[0], kernel_size=(1,5), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
     
    model.add(LearnedConv2D(
        gamma_trainable=trainable, 
        filters=ofmap[1], kernel_size=(1,5), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(LearnedConv2D(
        gamma_trainable=trainable,
        filters=ofmap[2], kernel_size=(1,5), padding='valid',
        dilation_rate=(1,1), input_shape = (1, in_shape, 4)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 64
    output_channel = input_channel*2
    
    model.add(LearnedConv2D(
        gamma_trainable=trainable,
        filters=ofmap[3], kernel_size=(1,9), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//2, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(LearnedConv2D(
        gamma_trainable=trainable,
        filters=ofmap[4], kernel_size=(1,9), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//2, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[5], 
        kernel_size=(1,5), 
        padding='valid', 
        strides=2))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 128
    output_channel = input_channel*2
    
    model.add(LearnedConv2D(
        gamma_trainable=trainable,
        filters=ofmap[6], kernel_size=(1,17), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//4, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(LearnedConv2D(
        gamma_trainable=trainable,
        filters=ofmap[7], kernel_size=(1,17), padding='same', 
        dilation_rate=(1,1), input_shape = (1, in_shape//4, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(filters=ofmap[8], 
        kernel_size=(1,5), 
        padding='valid', 
        strides=4))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(ofmap[9]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(ofmap[10]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(ofmap[11]))

    model.summary()
    
    return model

def TEMPONet_learned(width_mult, in_shape, dil_list, ofmap=[]):
    
    rf = [5, 9, 17]
   
    if not ofmap:
        ofmap = [
                32, 32, 62,
                63, 64, 128,
                91, 35, 43,
                64, 56, 1
                ]

    input_channel = width_mult * 32
    output_channel = input_channel * 2

    model = Sequential()
    
    model.add(layers.Conv2D(
        filters=ofmap[0], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[0])), 
        padding='same', dilation_rate=(1,dil_list[0]), 
        input_shape = (1, in_shape, 4)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[1], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[1])), 
        padding='same', dilation_rate=(1,dil_list[1]), 
        input_shape = (1, in_shape, 32)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
     
    model.add(layers.ZeroPadding2D(padding=((0, 0), (4, 0)))) 
    model.add(layers.Conv2D(
        filters=ofmap[2], 
        kernel_size=(1,math.ceil(rf[0]/dil_list[2])), 
        padding='valid', dilation_rate=(1,dil_list[2]), 
        input_shape = (1, in_shape+4, 32))) 
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 64
    output_channel = input_channel*2
    
    model.add(layers.Conv2D(
        filters=ofmap[3], 
        kernel_size=(1,math.ceil(rf[1]/dil_list[3])), 
        padding='same', dilation_rate=(1,dil_list[3]), 
        input_shape = (1, in_shape/2 + 8, 64)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[4], 
        kernel_size=(1,math.ceil(rf[1]/dil_list[4])), 
        padding='same', dilation_rate=(1,dil_list[4]), 
        input_shape = (1, in_shape/2 + 8, 64)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.ZeroPadding2D(padding=((0, 0), (4, 0))))
    model.add(layers.Conv2D(
        filters=ofmap[5], 
        kernel_size=(1,5), padding='valid', 
        strides=2, input_shape = (1, in_shape/2 + 4, 64)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    input_channel = width_mult * 128
    output_channel = input_channel*2
    
    model.add(layers.Conv2D(
        filters=ofmap[6], 
        kernel_size=(1,math.ceil(rf[2]/dil_list[5])), 
        padding='same', dilation_rate=(1,dil_list[5]), 
        input_shape = (1, in_shape/8 + 16, 128)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(
        filters=ofmap[7], 
        kernel_size=(1,math.ceil(rf[2]/dil_list[6])), 
        padding='same', dilation_rate=(1,dil_list[6]), 
        input_shape = (1, in_shape/8 + 16, 128)))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.ZeroPadding2D(padding=((0, 0), (5, 0))))
    model.add(layers.Conv2D(
        filters=ofmap[8], 
        kernel_size=(1,5), padding='valid',
        strides=4, input_shape = (1, in_shape/8 + 5, 128)))
    model.add(layers.AveragePooling2D(pool_size=(1,2), strides=2, padding='valid'))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(ofmap[9]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(ofmap[10]))
    model.add(layers.Activation('relu'))
    model.add(layers.BatchNormalization())
     
    model.add(layers.Dense(ofmap[11]))
   
    model.summary()
    
    return model
