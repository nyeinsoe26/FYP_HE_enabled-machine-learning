# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 02:31:28 2020

@author: nyein
"""

import numpy as np
import argparse
from phe import paillier
import socket


# def main(num_clients):
#     print("printing num_clients: {}".format(num_clients))
    
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.bind((socket.gethostname(), 1234))
#     s.listen(5)

#     while True:
#         # now our endpoint knows about the OTHER endpoint.
#         clientsocket, address = s.accept()
#         print(f"Connection from {address} has been established.")    
    
    
    
    
    
    
    
    
    
    
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("num_clients",help = "The total number of clients participating in this federated network")
#     args = parser.parse_args()
#     main(args.num_clients)   
import socket
import numpy as np
import pickle
import sys

def federated_averaging(input_list):
    num_clients = len(input_list)
    total = 0
    for i in range(num_clients):
        total = total + input_list[i]
    average = total*(1/num_clients)
    print("Averaging is done")
    return average
    
def sendDatasetToClient(dataset,data_type,s):
    print("sending "+data_type)
    data_tosend = pickle.dumps(dataset)
    s.send(bytes("msg_length:{}".format(len(data_tosend)),"utf-8"))
    s.send(data_tosend)

def exportNdarray(dataset, data_type,client_num):
    file_name =  "W_"+data_type+"_enc_Part"+str(client_num)
    np.save(file_name,dataset,allow_pickle = True)
    print("saving"+file_name)    
    
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 9999))
s.listen(5)
testing = []

print("Server has been launched")
#------------------------testing server-client-------------   
datasets = []

while True:
    
    clientsocket,address = s.accept()
    print(f"Connection from {address} has been establisted.")
    outermost_loopCount = 0 
    msg_len = 0    

    while True: 
            
        while msg_len==0:
            temp_len = clientsocket.recv(1234)
            temp_len = temp_len.decode("utf-8")
            if temp_len[0:10]=="msg_length":
                msg_len = int(temp_len[11:len(temp_len)])
                print("The length of the file to be received is {}".format(msg_len))
                break
            else:
                print("Invalid message length!!!")
        full_message = b''
        counter = 0
        while True:
            print("loop {}".format(counter))
        
            if len(full_message)==msg_len:
                print("unpickling full data at loop {}".format(counter))
                full_message = pickle.loads(full_message)
                msg_len = 0
                datasets.append(full_message)
                break
            
            else:
                msg = clientsocket.recv(65536)
                full_message = full_message + msg
                print("printing length of currently received message {}".format(len(msg)))
            counter = counter+1
        print("printing the shape of received array {}".format(full_message.shape))
        outermost_loopCount= outermost_loopCount+1
        
        if outermost_loopCount==2:
            break
    conv1 = []
    fc2 = []    
    if len(datasets)==8:
        conv1 = datasets[::2]
        fc2 = datasets[1::2]
        conv1_avg = federated_averaging(conv1)
        print("printing shape of average conv1 {}".format(conv1_avg.shape))
        np.save("conv1_avg",conv1_avg,allow_pickle = True)
        fc2_avg =  federated_averaging(fc2)
        print("printing shape of average fc2 {}".format(fc2_avg.shape))
        np.save("fc2_avg",fc2_avg,allow_pickle = True)
        clientsocket.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    