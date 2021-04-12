# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:36:50 2020

@author: MatteoRisso
"""

import numpy as np
import config as cf
import re
import utils
import math
import tensorflow as tf
from preprocessing_WordPTB import get_batch
from preprocessing_CharPTB import get_batch as get_batch_c



def train_Nottingham(model, ep, X_train, X_valid, X_test, train_losses, valid_losses, test_losses):
    epoch_losses = list()
    epoch_losses_valid = list()
    epoch_losses_test = list()

    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx].astype('float32')
        x_train, y_train = (data_line[:-1]), (data_line[1:])

        loss = model.train_on_batch(
           x = x_train.reshape(1, 1,x_train.shape[0] , cf.n_classes),
           y = y_train.reshape(1, 1,x_train.shape[0] , cf.n_classes))
        
        if ep > 8:
            if math.isnan(loss):
                tf.keras.models.save_model(model, 'model_'+str('nan')+'.h5')
                break
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
            x = x_valid.reshape(1, 1,x_valid.shape[0] , cf.n_classes),
            y = y_valid.reshape(1, 1,x_valid.shape[0] , cf.n_classes))
        epoch_losses_valid.append(valid_loss / x_valid.shape[0])

    train_losses.append(sum(epoch_losses) / (1.0 * len(epoch_losses)))
    valid_losses.append(sum(epoch_losses_valid) / (1.0 * len(epoch_losses_valid)))
    print("Epoch : {e} \t Loss : {l} \t Val_Loss : {v}".format(e=ep, l=train_losses[ep], v=valid_losses[ep]))

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

#@tf.function
def train_Word_PTB(model, ep, n_words, X_train, X_valid, X_test, train_losses, valid_losses, test_losses, warmup=False):
    
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=cf.lr, clipnorm=0.35)
    optimizer = tf.keras.optimizers.Adam(learning_rate=cf.lr, clipnorm=0.35)
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            final_target = tf.reshape(y[:, 0, eff_history:], (-1,))
            final_output = tf.reshape(logits[:, 0, eff_history:, :], (-1, n_words))
            #loss_value = loss_fn(y, logits)
            pred_loss = loss_fn(final_target, final_output)
            reg_loss = tf.reduce_sum(model.losses)
            if warmup:
                loss = pred_loss
            else:
                loss = pred_loss + reg_loss
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss
    
    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        final_target = tf.reshape(y[:, 0, eff_history:], (-1,))
        final_output = tf.reshape(val_logits[:, 0, eff_history:, :], (-1, n_words))
        #loss_value = loss_fn(y, val_logits)
        loss_value = loss_fn(final_target, final_output)
        return loss_value
    
    
    epoch_losses = list()
    epoch_losses_valid = list()
    epoch_losses_test = list()
    
    processed_data_size_valid = 0
    processed_data_size_test = 0
    total_loss = 0
    
    metrics_names = ['loss', 'val_loss']
    #progBar = tf.keras.utils.Progbar(X_train.shape[1], stateful_metrics=metrics_names)
    
    for batch_idx, i in enumerate(range(0, X_train.shape[1] - 1, cf.validseqlen)):
        if i + cf.seqlen - cf.validseqlen >= X_train.shape[1] - 1:
            continue
        # data.shape = targets.shape = (16, variable len)
        data, targets = get_batch(X_train, i, cf.seqlen)
        
        # Discard the effective history part
        eff_history = cf.seqlen - cf.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        #targets = targets[:, eff_history:]
        #final_output = np.reshape(output[:, eff_history:], (-1, n_words))
        
        
        loss = train_step(data.reshape(data.shape[0], 1, data.shape[1]),
            targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        values=[('train_loss',loss)]
        #progBar.update(batch_idx*cf.batch_size, values=values) 

        # loss = model.train_on_batch(
        #     x = data.reshape(data.shape[0], 1, data.shape[1]),
        #     y = targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        
        epoch_losses.append(loss)
        
        total_loss += loss
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            cur_loss = total_loss / 100
            #elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                ep, batch_idx, X_train.shape[1] // 40, cf.lr, cur_loss, math.exp(1)))
            total_loss = 0

            #start_time = time.time()
        
    train_losses.append(sum(epoch_losses) / (1.0 * len(epoch_losses)))
    
    for i in range(0, X_valid.shape[1] - 1, cf.validseqlen):
        if i + cf.seqlen - cf.validseqlen >= X_valid.shape[1] - 1:
            continue
        data, targets = get_batch(X_valid, i, cf.seqlen)
        
        # Discard the effective history part
        eff_history = cf.seqlen - cf.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        #targets = targets[:, eff_history:]
        #final_output = np.reshape(output[:, eff_history:], (-1, n_words))
        
        valid_loss = test_step(
            data.reshape(data.shape[0], 1, data.shape[1]),
            targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        # valid_loss = model.test_on_batch(
        #     x = data.reshape(data.shape[0], 1, data.shape[1]),
        #     y = targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        epoch_losses_valid.append((data.shape[1] - eff_history) * valid_loss)

        processed_data_size_valid += data.shape[1] - eff_history
    
    valid_losses.append(sum(epoch_losses_valid) / (1.0 * processed_data_size_valid))
    print("Epoch : {e} \t Loss : {l} \t Val_Loss : {v}".format(e=ep, l=train_losses[ep], v=valid_losses[ep]))

    for i in range(0, X_test.shape[1] - 1, cf.validseqlen):
        if i + cf.seqlen - cf.validseqlen >= X_test.shape[1] - 1:
            continue
        data, targets = get_batch(X_test, i, cf.seqlen)
        
        # Discard the effective history part
        eff_history = cf.seqlen - cf.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        #targets = targets[:, eff_history:]
        #final_output = np.reshape(output[:, eff_history:], (-1, n_words))
        
        test_loss = test_step(
            data.reshape(data.shape[0], 1, data.shape[1]),
            targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        # test_loss = model.test_on_batch(
        #     x = data.reshape(data.shape[0], 1, data.shape[1]),
        #     y = targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        epoch_losses_test.append((data.shape[1] - eff_history) * test_loss)

        processed_data_size_test += data.shape[1] - eff_history
    
    test_losses.append(sum(epoch_losses_test) / (1.0 * processed_data_size_test))
    print("Test loss : {}".format(test_losses[ep]))

    return train_losses[ep], valid_losses[ep], test_losses[ep]

def train_Char_PTB(model, ep, n_chars, X_train, X_valid, X_test, train_losses, valid_losses, test_losses, warmup=False):
    
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=cf.lr, clipnorm=0.15)
    #optimizer = tf.keras.optimizers.Adamax(learning_rate=lr, clipnorm=0.15)
    
    
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            #loss_value = loss_fn(y, logits)
            pred_loss = loss_fn(y, logits)
            reg_loss = tf.reduce_sum(model.losses)
            if warmup:
                loss = pred_loss
            else:
                loss = pred_loss + reg_loss
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss
    
    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        return loss_value, val_logits
    
    
    epoch_losses = list()
    epoch_losses_valid = list()
    epoch_losses_test = list()
    
    processed_data_size_valid = 0
    processed_data_size_test = 0
    
    metrics_names = ['loss', 'val_loss']
    progBar = tf.keras.utils.Progbar(X_train.shape[1], stateful_metrics=metrics_names)
    
    for batch_idx, i in enumerate(range(0, X_train.shape[1] - 1, cf.validseqlen)):
        if i + cf.seqlen - cf.validseqlen >= X_train.shape[1] - 1:
            continue
        # data.shape = targets.shape = (16, variable len)
        data, targets = get_batch_c(X_train, i, cf.seqlen)
        
        # Discard the effective history part
        eff_history = cf.seqlen - cf.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        targets = targets[:, eff_history:]
        #final_output = np.reshape(output[:, eff_history:], (-1, n_words))
        
        
        loss = train_step(data.reshape(data.shape[0], 1, data.shape[1]),
            targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        values=[('train_loss',loss)]
        progBar.update(batch_idx*cf.batch_size, values=values) 

        # loss = model.train_on_batch(
        #     x = data.reshape(data.shape[0], 1, data.shape[1]),
        #     y = targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        
        epoch_losses.append(loss)
    train_losses.append(sum(epoch_losses) / (1.0 * len(epoch_losses)))
    
    for i in range(0, X_valid.shape[1] - 1, cf.validseqlen):
        if i + cf.seqlen - cf.validseqlen >= X_valid.shape[1] - 1:
            continue
        data, targets = get_batch_c(X_valid, i, cf.seqlen)
        
        # Discard the effective history part
        eff_history = cf.seqlen - cf.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        targets = targets[:, eff_history:]
        #final_output = np.reshape(output[:, eff_history:], (-1, n_words))
        
        valid_loss, val_logits = test_step(
            data.reshape(data.shape[0], 1, data.shape[1]),
            targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        # valid_loss = model.test_on_batch(
        #     x = data.reshape(data.shape[0], 1, data.shape[1]),
        #     y = targets.reshape(targets.shape[0], 1, targets.shape[1]))
        val_logits = tf.reshape(val_logits, [-1, n_chars])
        epoch_losses_valid.append(val_logits.shape.as_list()[0] * valid_loss)

        processed_data_size_valid += val_logits.shape.as_list()[0]
    
    valid_losses.append(sum(epoch_losses_valid) / (1.0 * processed_data_size_valid))
    print("Epoch : {e} \t Loss : {l} \t Val_Loss : {v}".format(e=ep, l=train_losses[ep], v=valid_losses[ep]))

    for i in range(0, X_test.shape[1] - 1, cf.validseqlen):
        if i + cf.seqlen - cf.validseqlen >= X_test.shape[1] - 1:
            continue
        data, targets = get_batch_c(X_test, i, cf.seqlen)
        
        # Discard the effective history part
        eff_history = cf.seqlen - cf.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        targets = targets[:, eff_history:]
        #final_output = np.reshape(output[:, eff_history:], (-1, n_words))
        
        test_loss, test_logits = test_step(
            data.reshape(data.shape[0], 1, data.shape[1]),
            targets.reshape(targets.shape[0], 1, targets.shape[1]))
        
        # test_loss = model.test_on_batch(
        #     x = data.reshape(data.shape[0], 1, data.shape[1]),
        #     y = targets.reshape(targets.shape[0], 1, targets.shape[1]))
        test_logits = tf.reshape(test_logits, [-1, n_chars])
        epoch_losses_test.append(test_logits.shape.as_list()[0] * test_loss)

        processed_data_size_test += test_logits.shape.as_list()[0]
    
    test_losses.append(sum(epoch_losses_test) / (1.0 * processed_data_size_test))
    print("Test loss : {}".format(test_losses[ep]))

    return train_losses[ep], valid_losses[ep], test_losses[ep]

def warmup_Nottingham(model, epochs_num, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Nottingham(model, ep, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)

        if ep >= 2:
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

    return

def warmup_SeqMNIST(model, epochs_num, X_train, y_train, early_stop, checkpoint):
    hist = model.fit(
            x=X_train.reshape(-1, 1, X_train.shape[-1], 1),
            y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
            validation_split=0.1, verbose=1, 
            batch_size= cf.batch_size, epochs=epochs_num,
            callbacks = [early_stop, checkpoint])
    return

def warmup_Word_PTB(model, epochs_num, n_words, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Word_PTB(model, ep, n_words, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses, warmup=True)

        if ep >= 2:
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

    return

def warmup_Char_PTB(model, epochs_num, n_words, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Char_PTB(model, ep, n_words, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses, warmup=True)

        if ep >= 2:
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

    return

def train_gammas_Nottingham(model, epochs, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0
    gamma = dict()

    for ep in range(epochs):
        tr_l, v_l, t_l = train_Nottingham(model, ep, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)

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
                utils.save_dil_fact(cf.saving_path, gamma)

            else:
                wait += 1
                print("Val loss did not improve from {}".format(best_v_l))
                print("Iter for at least {} epochs".format(patience - wait))

        if wait >= patience:
            print("Early Stop")
            break

        ep += 1

def train_gammas_Word_PTB(model, epochs, n_words, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0
    gamma = dict()
    
    if cf.hyst == 1:
        alpha = dict()

    for ep in range(epochs):
        tr_l, v_l, t_l = train_Word_PTB(model, ep, n_words, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses)
        

        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()
        for name, weight in zip(names, weights):
            if re.search('learned_conv2d.+_?[0-9]/gamma', name): 
                gamma[name] = weight

            if cf.hyst == 1 and re.search('learned_conv2d.+_?[0-9]/alpha', name):
                alpha[name] = weight

        print(gamma)
        print(alpha)

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
                    #gamma[name] = weight
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

def train_gammas_Char_PTB(model, epochs, n_words, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0
    gamma = dict()

    for ep in range(epochs):
        tr_l, v_l, t_l = train_Char_PTB(model, ep, n_words, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses)
        

        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()
        for name, weight in zip(names, weights):
            if re.search('learned_conv2d.+_?[0-9]/gamma', name): 
                gamma[name] = weight

        print(gamma)

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
                    #gamma[name] = weight
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

def train_gammas_SeqMNIST(model, X_train, y_train, early_stop, save_gamma, exp_str): 
    hist = model.fit(
            x=X_train.reshape(-1, 1, X_train.shape[-1], 1),
            y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
            validation_split=0.1, verbose=1, 
            batch_size= cf.batch_size, epochs=cf.epochs,
            callbacks = [early_stop, save_gamma, exp_str])
    return

def retrain_Nottingham(model, epochs_num, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Nottingham(model, ep, X_train, X_valid, X_test, train_losses, valid_losses, test_losses)

        if ep >= 10:
            if np.less(v_l, best_v_l):
                print("Validation loss improved from {} to {}".format(best_v_l, v_l))
                best_v_l = v_l
                print("Save model to {}".format(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5'))
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

    return best_t_l

def retrain_Word_PTB(model, epochs_num, n_words, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Word_PTB(model, ep, n_words, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses)

    
        if np.less(v_l, best_v_l):
            print("Validation loss improved from {} to {}".format(best_v_l, v_l))
            best_v_l = v_l
            print("Save model to {}".format(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5'))
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

    return best_t_l

def retrain_Char_PTB(model, epochs_num, n_words, X_train, X_valid, X_test):
    train_losses = list()
    valid_losses = list()
    test_losses = list()

    best_v_l = np.Inf
    patience = 30
    wait = 0

    for ep in range(epochs_num):
        tr_l, v_l, t_l = train_Char_PTB(model, ep, n_words, X_train, X_valid, X_test, 
                                      train_losses, valid_losses, test_losses)

    
        if np.less(v_l, best_v_l):
            print("Validation loss improved from {} to {}".format(best_v_l, v_l))
            best_v_l = v_l
            print("Save model to {}".format(cf.saving_path+'autodil/trained_weights_warmup'+str(cf.warmup)+'.h5'))
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

    return best_t_l

def retrain_SeqMNIST(model, epochs_num, X_train, y_train, X_test, y_test, early_stop, checkpoint):
    hist = model.fit(
        x=X_train.reshape(-1, 1, X_train.shape[-1], 1),
        y=y_train.reshape(y_train.shape[0], 1), shuffle=True, 
        validation_data=(X_test.reshape(-1, 1, X_test.shape[-1], 1), y_test.reshape(y_test.shape[0], 1)), 
        verbose=1, 
        batch_size= cf.batch_size, epochs=cf.epochs,
        callbacks = [early_stop, checkpoint])
    return


