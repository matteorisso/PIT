# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:31:50 2020

@author: MatteoRisso
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def NLL(y_true, y_pred):

    return -tf.linalg.trace(
        tf.matmul(
            tf.cast(y_true, dtype='float32'),
            tf.transpose(tf.cast(tf.math.log(y_pred + 1e-10), dtype='float32'), [0, 1, 3, 2])
            ) +
        tf.matmul(
            tf.cast((1 - y_true), dtype='float32'),
            tf.transpose(tf.cast(tf.math.log(1 - y_pred + 1e-10), dtype='float32'), [0, 1, 3, 2])
            )
        )

def accuracy(y_true, y_pred):
            # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
            if K.ndim(y_true) == K.ndim(y_pred):
                y_true = K.squeeze(y_true, -1)
            # convert dense predictions to labels
            y_pred_labels = K.argmax(y_pred, axis=-1)
            y_pred_labels = K.cast(y_pred_labels, K.floatx())
            return K.cast(K.equal(y_true, y_pred_labels), K.floatx())
