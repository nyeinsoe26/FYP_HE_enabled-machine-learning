# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 02:13:47 2020

@author: nyein
"""

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from pathlib import Path
import numpy as np
import pandas as pd

from keras.datasets import cifar10
from keras.datasets import mnist
from keras.datasets import fashion_mnist

def load_dataset(data):
    if data=="cifar10":
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    elif data=="fashion_mnist":
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
    else:
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
    
    print("x_train shape: {}, y_train shape: {}".format(x_train.shape,y_train.shape))
    print("x_train type: {}, y_train type: {}".format(type(x_train),type(y_train)))
    
    return x_train,y_train,x_test,y_test

def normalize_data(x_train,y_train,x_test,y_test):
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = x_train/255
    x_test = x_test/255
    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test,10)
    print("x_train shape: {}, y_train shape: {}".format(x_train.shape,y_train.shape))
    print("x_train type: {}, y_train type: {}".format(type(x_train),type(y_train)))
    return x_train,y_train,x_test,y_test

def in_order_sampling(x_train,y_train,num_clients):
    clients_data = []
    clients = []
    
    partition_length = int((len(x_train))/num_clients)
    
    for i in range(num_clients):
        clients_data.append(x_train[0:partition_length])
        clients_data.append(y_train[0:partition_length])
        clients.append(clients_data)
        
        x_train = x_train[partition_length:len(x_train)]
        y_train = y_train[partition_length:len(y_train)]
        
        clients_data = []
    return clients

def repeatedSampling(x_train,y_train,num_clients):
 
    partition_length = int((len(x_train))/num_clients)
    x_train,y_train = shuffle(x_train,y_train)
    clients_data = []
    clients = []
    
    seed(1)
    for i in range(num_clients):
        temp_x = np.zeros((partition_length,32,32,3))
        temp_y = np.zeros((partition_length,10))
        for index in range(partition_length):
            randnum = randint(0,len(x_train)-1)
            temp_x[index] = x_train[randnum]
            temp_y[index] = y_train[randnum]
            
        clients_data.append(temp_x)
        clients_data.append(temp_y)
        clients.append(clients_data)
        
        x_train,y_train = shuffle(x_train,y_train)
        
        clients_data = []
    return clients

def non_repeatedSampling(x_train,y_train,num_clients):
 
    partition_length = int((len(x_train))/num_clients)
    x_train,y_train = shuffle(x_train,y_train)
    clients_data = []
    clients = []
    
    seed(1)
    for i in range(num_clients):
        clients_data.append(x_train[0:partition_length])
        clients_data.append(y_train[0:partition_length])
        clients.append(clients_data)
        
        x_train = x_train[partition_length:len(x_train)]
        y_train = y_train[partition_length:len(y_train)]
        x_train,y_train = shuffle(x_train,y_train)
        
        clients_data = []
    return clients


x_train_c,y_train_c,x_test_c,y_test_c = load_dataset("cifar10")
x_train_m,y_train_m,x_test_m,y_test_m = load_dataset("mnist")
x_train_f,y_train_f,x_test_f,y_test_f = load_dataset("fashion_mnist")

#training specifications
batch_size = 1000
NUM_CHANNELS = 3 #RGB image
conv_fm1 = 50
conv_fm2=60
NUM_CLASSES=10