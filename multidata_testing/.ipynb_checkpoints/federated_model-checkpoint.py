#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from pathlib import Path
import numpy as np
import pandas as pd
import client_util


# In[2]:


#training specifications
conv_fm1 = 50
conv_fm2=60
NUM_CLASSES=10


# In[3]:


def model(x,y_actual,NUM_CHANNELS,keep_prob):
    print("=================Setting up Cifar10 model=================")
    #First convolution layer
    with tf.variable_scope('conv_1',reuse=tf.AUTO_REUSE):
        W1 = tf.Variable(tf.truncated_normal([9, 9, NUM_CHANNELS, conv_fm1], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9)), name='weights_1')
        b1 = tf.Variable(tf.zeros([conv_fm1]), name='biases_1')
        conv_1 = tf.nn.relu(tf.nn.conv2d(x, W1, [1, 1, 1, 1], padding='SAME') + b1)
        pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_1')
        print("W1_shape: {}, b1_shape: {}".format(tf.shape(W1),tf.shape(b1)))
    
    #2nd convolution layer
    with tf.variable_scope('conv_2',reuse=tf.AUTO_REUSE):
        W2 = tf.Variable(tf.truncated_normal([5, 5, conv_fm1, conv_fm2], stddev=1.0 / np.sqrt(5 * 5 * conv_fm1)), name='weights_2')
        b2 = tf.Variable(tf.zeros([conv_fm2]), name='biases_2')
        conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='SAME') + b2)
        pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool_2')
        print("W2_shape: {}, b2_shape: {}".format(tf.shape(W2),tf.shape(b2)))
        
    print("dim1 shape: {}, dim2 shape: {}, dim3 shape:{} ".format(pool_2.get_shape()[1].value,pool_2.get_shape()[2].value,pool_2.get_shape()[3].value))
    #flatten pool_2
    dim2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
    pool_2_flat = tf.reshape(pool_2, [-1, dim2])

    #fully connected layer
    with tf.variable_scope('fc_1',reuse = tf.AUTO_REUSE):
        W_fc1 = tf.Variable(tf.truncated_normal([dim2, 300], stddev=1.0 / np.sqrt(dim2)), name='weights_3')
        b_fc1 = tf.Variable(tf.zeros([300]), name='biases_3')
        fc_1 = tf.nn.relu(tf.matmul(pool_2_flat, W_fc1) + b_fc1)
        print("W_fc1_shape: {}, b_fc1_shape: {}".format(tf.shape(W_fc1),tf.shape(b_fc1)))
    
    #add dropout layer
    fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    #predict output
    W_fc2 = tf.Variable(tf.truncated_normal([300, NUM_CLASSES], stddev=1.0 / np.sqrt(300)), name='weights_4')
    b_fc2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
    y_pred = tf.matmul(fc_1_drop, W_fc2) + b_fc2
    print("W_fc2_shape: {}, b_fc2_shape: {}".format(tf.shape(W_fc2),tf.shape(b_fc2)))
    return y_pred


# In[4]:


#

