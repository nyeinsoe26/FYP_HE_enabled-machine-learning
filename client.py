# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 02:32:09 2020

@author: nyein
"""
import socket
import argparse
import numpy as np
import pickle
import encrypt_decrypt
import os.path
from os import path
from phe import paillier

if path.exists("keys.npy"):
   key_pair = np.load("keys.npy",allow_pickle = True)
   public_key = key_pair[0]
   private_key = key_pair[1]
else:
   keys = []
   public_key, private_key = paillier.generate_paillier_keypair()
   keys.append(public_key)
   keys.append(private_key)
   np.save("keys",keys,allow_pickle=True)

def load_dataset(data_type,client_num):
    if data_type=="conv1":
        shape_ = [5,5,1,5]
    elif data_type=="fc2":
        shape_ = [100,10]
    else:
        shape_ = [5*13*13,100]
    file_name = "W_"+data_type+"_repeatedSampling_Part"+str(client_num)+".txt"
    dataset = np.loadtxt(file_name,dtype=np.float64).reshape(shape_)
    print("datasets loaded")
    return dataset
 
        
    
    
def encrypt_givenDatasets(dataset):
    encrypted_data = encrypt_decrypt.encrypt_data(dataset,public_key)
    print("printing shape of encrypted data {}".format(encrypted_data.shape))
    return encrypted_data


    
def exportNdarray(dataset, data_type,client_num):
    file_name =  "W_"+data_type+"_enc_Part"+str(client_num)
    np.save(file_name,dataset,allow_pickle = True)
    print("saving"+file_name)

def load_encryptedDatasets(data_type,client_num):
    file_name =  "W_"+data_type+"_enc_Part"+str(client_num)+".npy"
    dataset = np.load(file_name,allow_pickle=True)
    return dataset
    
def sendDatasetToAggre(dataset,data_type,s):
    print("sending "+data_type)
    data_tosend = pickle.dumps(dataset)
    s.send(bytes("msg_length:{}".format(len(data_tosend)),"utf-8"))
    s.send(data_tosend)
    

def main(client_num):

    encrypt_done = False
    send_to_aggregator = False
    
    while encrypt_done==False:
        choice = input("""
                    Enter choice:
                    1. Encrypt data
                    2. Send encrypted data to Aggregator
                    """)
        if choice=="1":
            W_conv1 = load_dataset("conv1",client_num)
            W_conv1_enc = encrypt_givenDatasets(W_conv1)
            exportNdarray(W_conv1_enc, "conv1",client_num)
            
            W_fc2 = load_dataset("fc2",client_num)
            W_fc2_enc = encrypt_givenDatasets(W_fc2)
            exportNdarray(W_fc2_enc, "fc2",client_num)
            
            proceed = input("Proceed to send data to aggregator (Y/N): ")
            if proceed.lower()=='n':
                print("Exiting client!!!")
            else:
                send_to_aggregator=True
            encrypt_done = True    
        elif choice=="2":
            encrypt_done = True  
            send_to_aggregator = True            
        
    if send_to_aggregator==True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((socket.gethostname(), 9999))
        
        W_conv1_enc = load_encryptedDatasets("conv1",client_num)
        print(W_conv1_enc[0,0])
        sendDatasetToAggre(W_conv1_enc,"conv1",s)
        W_fc2_enc = load_encryptedDatasets("fc2",client_num)
        sendDatasetToAggre(W_fc2_enc,"fc2",s)
        
        while True:
            #wait for return file from server then decrypt
            if path.exists("conv1_avg.npy") and path.exists("fc2_avg.npy"):
                choice = input("Average Data ready. Proceed to download (Y/N)?")
                if choice.lower()=='n':
                    break
                else:
                    conv1_avg = np.load("conv1_avg.npy",allow_pickle=True)
                    fc2_avg = np.load("fc2_avg.npy",allow_pickle= True)
                    print("Data downloaded and proceeding to decrypt")
                    conv1_avg_dec = encrypt_decrypt.decrypt_data(conv1_avg,private_key)
                    fc2_av_dec = encrypt_decrypt.decrypt_data(fc2_avg,private_key)
                    print("saving files")
                    np.save("conv1_avg_dec",conv1_avg_dec,allow_pickle=True)
                    np.save("fc2_av_dec",fc2_av_dec,allow_pickle=True)
                    break
       
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("client_num",help = "Index num of client")
    args = parser.parse_args()
    main(args.client_num)   


