# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 02:31:45 2020

@author: nyein
"""

import numpy as np
import pickle 
from phe import paillier

def encrypt_data(dataset,public_key):
    num_dim = dataset.ndim
    
    if num_dim==2:
        encrypted_list = [[public_key.encrypt(x) for x in row] for row in dataset]
        print("Data is encrypted!!!")
    else:
        encrypted_list = [[[[public_key.encrypt(first) for first in second]for second in third]for third in fourth]for fourth in dataset]
        print("Conv Data is encrypted!!!")

    encrypted_array = np.array(encrypted_list)
    return encrypted_array

def decrypt_data(dataset,private_key):
    num_dim = dataset.ndim
    
    if num_dim==2:
        decrypted_list = [[private_key.decrypt(x) for x in row] for row in dataset]
        print("Data is decrypted!!!")
    else:
        decrypted_list = [[[[private_key.decrypt(first) for first in second]for second in third]for third in fourth]for fourth in dataset]
        print("Conv Data is decrypted!!!")

    decrypted_array = np.array(decrypted_list)
    return decrypted_array