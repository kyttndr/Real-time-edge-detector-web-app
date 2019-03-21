import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import backend as K

def Mongo_loss(y_true, y_pred):
    beta = 0.6
    smooth = 0.0001
    y_true_non_edge = tf.math.subtract(1., K.flatten(y_true))
    y_pred_non_edge = tf.math.add(smooth, tf.math.subtract(1., K.flatten(y_pred)))
    y_true_edge = K.flatten(y_true)
    y_pred_edge = tf.math.add(smooth, K.flatten(y_pred))
    y_edge = tf.math.multiply(y_true_edge, y_pred_edge)
    y_non_edge = tf.math.multiply(y_true_non_edge, y_pred_non_edge)
    y_edge_without_zeroes = tf.boolean_mask(y_edge, tf.not_equal(y_edge, 0))
    y_non_edge_without_zeroes = tf.boolean_mask(y_non_edge, tf.not_equal(y_non_edge, 0))
    return (-beta)*tf.math.cumsum(tf.math.log(y_edge_without_zeroes))[-1] - (1-beta)*tf.math.cumsum(tf.math.log(y_non_edge_without_zeroes))[-1]

def f1(y_true, y_pred):
    # https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    def recall(y_true, y_pred):
#         Recall metric.
#         Only computes a batch-wise average of recall.
#         Computes the recall, a metric for multi-label classification of
#         how many relevant items are selected.

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
#         Precision metric.
#         Only computes a batch-wise average of precision.
#         Computes the precision, a metric for multi-label classification of
#         how many selected items are relevant.
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def dice_coef(y_true, y_pred, smooth=0.5):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)



from keras.models import load_model
model = load_model('/home/ubuntu/Real-time-edge-detector-web-app/model.h5', custom_objects={"Mongo_loss": Mongo_loss, "K": K, "f1": f1, "tf": tf, "dice_coef_loss": dice_coef_loss})






