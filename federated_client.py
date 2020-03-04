import socket
import argparse
import numpy as np
import pickle
import encrypt_decrypt
import os.path
from os import path
from phe import paillier
import importlib

conv1_len = []
fc2_len = []
max_len_list = []
avg_data = []


if path.exists("keys.npy"):
   key_pair = np.load("keys.npy",allow_pickle = True)
   public_key = key_pair[0]
   private_key = key_pair[1]
   print("Existing Keys loaded")
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
    s.send(bytes("",'UTF-8'))
    s.send(bytes("msg_length:{}".format(len(data_tosend)),'UTF-8'))
    if data_type =="conv1":
        conv1_len.append(len(data_tosend))
    else:
        fc2_len.append(len(data_tosend))
    data_rdyTo_send = s.recv(1234)
    data_rdyTo_send = data_rdyTo_send.decode()
    if data_rdyTo_send=="ready":
        s.send(data_tosend)
    
def main(client_num):
    encrypt_done = False
    send_to_aggregator = False
    decrypt_done = False
    
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
        SERVER = "127.0.0.1"
        PORT = 8080
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((SERVER, PORT))
        
        #send conv
        W_conv1_enc = load_encryptedDatasets("conv1",client_num)
        sendDatasetToAggre(W_conv1_enc,"conv1",client)
        
        #send fc
        W_fc2_enc = load_encryptedDatasets("fc2",client_num)
        sendDatasetToAggre(W_fc2_enc,"fc2",client)
        
        
        full_fc = b''
        while True:
            importlib.reload(pickle)
            in_data =  client.recv(1024)
            in_data = in_data.decode()
            if in_data.lower()=="job completed":
                c = input("Average Data ready. Proceed to download (Y/N)?")
                if c.lower() == 'n':
                    break
                else:
                    
                    client.send(bytes("download data",'UTF-8'))
                    
                    num_files = 0
                    full_conv = b''
                    while True:
                        if num_files == 0:
                            message_length = 69920
                        else:
                            message_length = 552422
                        while True:
                            if len(full_conv)!=message_length:
                                msg = client.recv(65536)
                                full_conv = full_conv + msg
                            else:
                                full_conv = pickle.loads(full_conv)
                                avg_data.append(full_conv)
                                print("printing shape of average data: {}".format(full_conv.shape))
                                full_conv = b''
                                break
                        num_files = num_files+1
                        if num_files==2:
                            num_files = 0
                            break
            if len(avg_data)==2:
                break
        print("Data downloaded and proceeding to decrypt")
        conv1_avg_dec = encrypt_decrypt.decrypt_data(avg_data[0],private_key)
        fc2_av_dec = encrypt_decrypt.decrypt_data(avg_data[1],private_key)
        print("Exporting decrypted files")
        np.save("conv1_avg_dec",conv1_avg_dec,allow_pickle=True)
        np.save("fc2_av_dec",fc2_av_dec,allow_pickle=True)                    
           
               
    client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("client_num",help = "Index num of client")
    args = parser.parse_args()
    main(args.client_num)   