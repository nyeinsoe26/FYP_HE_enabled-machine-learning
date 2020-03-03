"""An MNIST classifier based on Cryptonets using convolutional layers. """
#importing libriaries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import numpy as np
import itertools
import tensorflow as tf
import model
import os
import pandas as pd
import encrypt_decrypt
from random import seed
from random import randint
from sklearn.utils import shuffle

from phe import paillier
public_key, private_key = paillier.generate_paillier_keypair()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mnist_util import load_mnist_data, \
    get_variable, \
    conv2d_stride_2_valid, \
    avg_pool_3x3_same_size, \
    get_train_batch

def sampleData_withRepeat(x,y, num_partition):
    partition_length = int((len(x))/num_partition)
    x,y = shuffle(x,y)
    x_partition = []
    y_partition = []
    seed(1)
    for i in range(num_partition):
        temp_x = np.zeros((partition_length,28,28,1))
        temp_y = np.zeros((partition_length,10))
        for index in range(partition_length):
            randnum = randint(0,len(x)-1)
            temp_x[index] = x[randnum]
            temp_y[index] = y[randnum]
        x_partition.append(temp_x)
        y_partition.append(temp_y)
        x,y = shuffle(x,y)
    return x_partition,y_partition

def sampleData_withNoRepeat(x,y,num_partition):
    partition_length = int((len(x))/num_partition)
    x,y = shuffle(x,y)
    x_partition = []
    y_partition = []
    for i in range(num_partition):
        x_partition.append(x[0:partition_length])
        y_partition.append(y[0:partition_length])

        x = x[partition_length:len(x)]
        y = y[partition_length:len(y)]
        x,y = shuffle(x,y)
    return x_partition,y_partition

def sampleData_inOrder(data,num_partition):
    partition_length = int((len(data))/num_partition)
    partition = []
    for i in range(num_partition):
        partition.append(data[0:partition_length])
        data = data[partition_length:len(data)]
    return partition
    
def squash_layers(index):
    tf.compat.v1.reset_default_graph()
    print("Squashing layers for part {0}".format(index))
    # Input from h_conv1 squaring
    x = tf.compat.v1.placeholder(tf.float32, [None, 13, 13, 5])

    # Pooling layer
    h_pool1 = avg_pool_3x3_same_size(x)  # To N x 13 x 13 x 5
    
    W_conv2_txt = "W_conv2_repeatedSampling_Part"+str(index)+".txt"
    # Second convolution
    W_conv2 = np.loadtxt(
        W_conv2_txt, dtype=np.float32).reshape([5, 5, 5, 50])
    h_conv2 = conv2d_stride_2_valid(h_pool1, W_conv2)

    # Second pooling layer.
    h_pool2 = avg_pool_3x3_same_size(h_conv2)

    # Fully connected layer 1
    # Input: N x 5 x 5 x 50
    # Output: N x 100
    W_fc1_txt = "W_fc1_repeatedSampling_Part"+str(index) + ".txt"
    W_fc1 = np.loadtxt(
        W_fc1_txt, dtype=np.float32).reshape([5 * 5 * 50, 100])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 50])
    pre_square = tf.matmul(h_pool2_flat, W_fc1)

    with tf.compat.v1.Session() as sess:
        x_in = np.eye(13 * 13 * 5)
        x_in = x_in.reshape([13 * 13 * 5, 13, 13, 5])
        W = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        squashed_file_name = "W_squash_repeatedSampling_Part"+str(index)+".txt"
        np.savetxt(squashed_file_name, W)
        print("Saved to", squashed_file_name)

        # Sanity check
        x_in = np.random.rand(100, 13, 13, 5)
        network_out = (sess.run([pre_square], feed_dict={x: x_in}))[0]
        linear_out = x_in.reshape(100, 13 * 13 * 5).dot(W)
        assert (np.max(np.abs(linear_out - network_out)) < 1e-5)

    print("Squashed layers")
	
def main(FLAGS):
    (x_train, y_train, x_test, y_test) = load_mnist_data()
    print("x_train: {0}, y_train:{1}, x_test: {2}, y_test: {3}".format(len(x_train),len(y_train),len(x_test),len(y_test)))
    print("printing originalshape: {}".format(y_train.shape))
    print("printing original type: {}".format(type(x_train)))
    print("printing original length: {}".format(len(x_train)))
    x_train_partition,y_train_partition = sampleData_withRepeat(x_train,y_train,4)
    x_test_partition,y_test_partition = sampleData_withRepeat(x_test,y_test,4)
    
    
    x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
    y_ = tf.compat.v1.placeholder(tf.float32, [None, 10])
    y_conv = model.cryptonets_model(x, 'train')
    

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(
            cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
    
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for index in range(4):
            print("Training part {0}".format(index+1))
            loss_values = []
            for i in range(FLAGS.train_loop_count):
                x_batch, y_batch = get_train_batch(i, FLAGS.batch_size, x_train_partition[index],
                                               y_train_partition[index])
                if i % 100 == 0:
                    t = time.time()
                    train_accuracy = accuracy.eval(feed_dict={
                        x: x_batch,
                        y_: y_batch
                    })
                    print('step %d, training accuracy %g, %g msec to evaluate' %
                          (i, train_accuracy, 1000 * (time.time() - t)))
                t = time.time()
                _, loss = sess.run([train_step, cross_entropy],
                                   feed_dict={
                                       x: x_batch,
                                       y_: y_batch
                                   })
                loss_values.append(loss)

                if i % 1000 == 999 or i == FLAGS.train_loop_count - 1:
                    test_accuracy = accuracy.eval(feed_dict={
                        x: x_test_partition[index],
                        y_: y_test_partition[index]
                    })
                    print('test accuracy %g' % test_accuracy)

            print("Training finished. Saving variables.")
            for var in tf.compat.v1.get_collection(
                    tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
                weight = (sess.run([var]))[0].flatten().tolist()
                filename = (str(var).split())[1].replace('/', '_')
                filename = filename + "_repeatedSampling_Part"+str(index+1)
                filename = filename.replace("'", "").replace(':0', '') + '.txt'

                print("saving", filename)
                np.savetxt(str(filename), weight)
    

    # Squash weights and save as W_squash.txt
    for index in range(4):
        print("testing")
        squash_layers(index+1)
        
    W_conv1 = load_dataset("conv1")
    W_conv1_encr = encrypt_givenDatasets(W_conv1)
    exportNdarray(W_conv1_encr, "conv1")
    

def load_dataset(data_type):
    if data_type=="conv1":
        shape_ = [5,5,1,5]
    elif data_type=="fc2":
        shape_ = [100,10]
    else:
        shape_ = [5*13*13,100]
    dataset = []
    for i in range(4):
        file_name = "W_"+data_type+"_repeatedSampling_Part"+str(i+1)+".txt"
        data = np.loadtxt(file_name,dtype=np.float64).reshape(shape_)
        print("printing original shape {}".format(data[i].shape))
        dataset.append(data)
        
    print("datasets loaded")
    return dataset
    
def encrypt_givenDatasets(dataset):
    encrypted_data = []
    for i in range(4):
        encrypted_data.append(encrypt_decrypt.encrypt_data(dataset[i],public_key))
        print("printing shape of encrypted data {}".format(encrypted_data[i].shape))
    return encrypted_data

    
def exportNdarray(dataset, data_type):
    for i in range(4):
        file_name =  "W_"+data_type+"_enc_Part"+str(i+1)
        np.save(file_name,dataset[i],allow_pickle = True)
        print("saving"+file_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_loop_count',
        type=int,
        default=5000,
        help='Number of training iterations')
    parser.add_argument(
        '--batch_size', type=int, default=50, help='Batch Size')
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
	
