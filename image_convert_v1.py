import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import load_model

batch_size = 1

def Mongo_Jr_loss(y_true, y_pred):
    beta = 0.75
    smooth1 = 0.001
    smooth2 = 0.001
    
    i = 0
    
    loss = 0
    
    bs = batch_size
    
    for i in range(0, bs):
        y_true_non_edge = tf.subtract(1., K.flatten(y_true[i]))
        y_pred_non_edge = tf.add(smooth1, tf.subtract(1., K.flatten(y_pred[i])))
        
        y_true_edge = K.flatten(y_true[i])
        y_pred_edge = tf.add(smooth1, K.flatten(y_pred[i]))
        
        y_edge = tf.multiply(y_true_edge, y_pred_edge)
        y_non_edge = tf.multiply(y_true_non_edge, y_pred_non_edge)
        
        y_edge_without_zeroes = tf.boolean_mask(y_edge, tf.not_equal(y_edge, 0))
        y_non_edge_without_zeroes = tf.boolean_mask(y_non_edge, tf.not_equal(y_non_edge, 0))
        
        loss = loss + (-beta)*tf.cumsum(tf.log(y_edge_without_zeroes))[-1] - (1-beta)*tf.cumsum(tf.log(y_non_edge_without_zeroes))[-1]
        i += 1

    return loss

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

model = load_model('/home/ubuntu/Real-time-edge-detector-web-app/model_37.h5', custom_objects={"Mongo_Jr_loss": Mongo_Jr_loss, "K": K, "f1": f1, "tf": tf, "batch_size": batch_size})
