import numpy as np
import argparse
from phe import paillier
import os
import pickle
import sys
import socket, threading
from threading import Thread
import importlib

clients = set()
clients_lock = threading.Lock()
num_messages = []
weights = []
conv1_avg = []
fc2_avg = []

def federated_averaging(input_list):
    num_clients = len(input_list)
    total = 0
    for i in range(num_clients):
        total = total + input_list[i]
    average = total*(1/num_clients)
    print("Averaging is done")
    return average
    
def listener(client, address):
    print("Accepted connection from: {}".format(address))
    with clients_lock:
        clients.add(client)
        print("printing number of clients connected so far: {}".format(len(clients))) 

        
    while True:
        msg_len = 0
        full_message = b''
        num_files = 0
        while True: 
            while msg_len==0:
                temp_len = client.recv(1234)
                temp_len = temp_len.decode()
                if temp_len[0:10]=="msg_length":
                    msg_len = int(temp_len[11:len(temp_len)])
                    print("The length of the file to be received is {}".format(msg_len))
                    client.send(bytes("ready",'UTF-8'))
                    break #break out of msg_len==0
            counter = 0 #to count number of loops taken to open full file
            while True:
                print("loop {}".format(counter))
                if len(full_message)==msg_len:
                    print("unpickling full data at loop {}".format(counter))
                    full_message = pickle.loads(full_message)
                    weights.append(full_message)
                    msg_len = 0
                    full_message = b''
                    break #break inner while True
                else:
                    msg = client.recv(65536)
                    full_message = full_message + msg
                    print("printing length of currently received message {}".format(len(msg)))
                counter = counter+1
            #reach here means received data fully and alr broke out of inner while=True loop
            num_files= num_files + 1
            if num_files==2:
                num_files = 0
                break  #break out of middle while = True loop
                
        if len(weights)==8:
            #perform averaging
            conv1 = weights[::2]
            fc2 = weights[1::2]
            temp = federated_averaging(conv1)
            conv1_avg.append(temp)
            print("printing shape of average conv1 {}".format(temp.shape))
        
            temp2 =  federated_averaging(fc2)
            fc2_avg.append(temp2)
            print("printing shape of average fc2 {}".format(temp2.shape))
        
            with clients_lock:
                for c in clients:
                    c.sendall(bytes("job completed",'UTF-8'))
        
        while True:
            download = client.recv(1234)
            download = download.decode()     
            if download.lower() =="download data":
                break
                
        print("sending average conv1")
        data_to_send = pickle.dumps(conv1_avg[0])
        print("printing length of average_conv1 {}".format(len(data_to_send)))
        client.send(data_to_send) 
        
        print("sending average fc2")
        fc_data = pickle.dumps(fc2_avg[0])
        print("printing length of average_fc2 {}".format(len(fc_data)))
        client.send(fc_data)
        while True:
            download = client.recv(1234)
            download = download.decode()     
            if download.lower() =="download data":
                break 
        # while True:
            # download = client.recv(1234)
            # download = download.decode()
            # if download.lower()=="download conv":                
                # print("sending average conv1")
                # data_to_send = pickle.dumps(conv1_avg[0])
                # client.send(data_to_send)
                # break
        # while True:
            
            # download = client.recv(1234)
            # download = download.decode()
            # if download.lower()=="download fc2":                
                # print("sending average fc2")
                # data_to_send = pickle.dumps(fc2_avg[0])
                # client.send(data_to_send)
                # print("fc2 sent")
                # break
 
                    
            

            

host = "127.0.0.1"
port = 8080

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((host,port))
s.listen(3)
th = []

while True:
    print("Server is listening for connections...")
    client, address = s.accept()
    th.append(Thread(target=listener, args = (client,address)).start())

s.close()