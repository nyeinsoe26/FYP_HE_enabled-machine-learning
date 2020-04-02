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
import os
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from sklearn.utils import shuffle
from random import seed
from random import randint

def read_clientDir(clientNum,cwd):
    folderPath = "client" + str(clientNum)
    folderDir = os.path.join(cwd,folderPath,'model_weights')
    os.chdir(folderDir)
    client_data = []
    conv1_W1 = np.load("conv_1_weights_1.npy",allow_pickle=True)
    conv1_b1 = np.load("conv_1_biases_1.npy",allow_pickle=True)
    conv2_W2 = np.load("conv_2_weights_2.npy",allow_pickle=True)
    conv2_b2 = np.load("conv_2_biases_2.npy",allow_pickle=True)
    W_fc1 = np.load("fc_1_weights_3.npy",allow_pickle=True)
    b_fc1 = np.load("fc_1_biases_3.npy",allow_pickle=True)
    W_fc2 = np.load("weights_4.npy",allow_pickle=True)
    b_fc2 = np.load("biases_4.npy",allow_pickle=True)
    client_data.append(conv1_W1)
    client_data.append(conv1_b1)
    client_data.append(conv2_W2)
    client_data.append(conv2_b2)
    client_data.append(W_fc1)
    client_data.append(b_fc1)
    client_data.append(W_fc2)
    client_data.append(b_fc2)
    os.chdir(cwd)
    print("printing length of the client_list: {}".format(len(client_data)))
    return client_data

def create_folder(folder_name,cwd):
    folderPath = folder_name
    folderDir = os.path.join(cwd,folderPath)
    if not os.path.isdir(folderDir):
        print('creating the {} folder'.format(folderPath))
        os.makedirs(folderDir)
    os.chdir(folderDir)
    
def change_dir(folder_name,cwd):
    folderPath = folder_name
    folderDir = os.path.join(cwd,folderPath)
    os.chdir(folderDir)
    
def save_clientData(avg_weights):
    np.save("conv1_W1_avg.npy",avg_weights[0],allow_pickle = True)
    np.save("conv1_b1_avg.npy",avg_weights[1],allow_pickle = True)
    np.save("conv2_W2_avg.npy",avg_weights[2],allow_pickle = True)
    np.save("conv2_b2_avg.npy",avg_weights[3],allow_pickle = True)
    np.save("fc1_W3_avg.npy",avg_weights[4],allow_pickle = True)
    np.save("fc1_b3_avg.npy",avg_weights[5],allow_pickle = True)
    np.save("w4_avg.npy",avg_weights[6],allow_pickle = True)
    np.save("b4_avg.npy",avg_weights[7],allow_pickle = True)
    
def load_dataset(data):
    if data=="cifar10":
        (x_train,y_train),(x_test,y_test) = cifar10.load_data()
    elif data=="fashion_mnist":
        (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
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

def repeatedSampling(x_train,y_train,num_clients,dataset):
 
    partition_length = int((len(x_train))/num_clients)
    x_train,y_train = shuffle(x_train,y_train)
    clients_data = []
    clients = []
    if dataset == "cifar10":
        img_width = 32
        img_height = 32
        num_channels = 3
    else:
        img_width = 28
        img_height = 28
        num_channels = 1
        
    seed(1)
    for i in range(num_clients):
        temp_x = np.zeros((partition_length,img_width,img_height,num_channels))
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

