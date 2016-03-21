# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:43:09 2016

@author: Abhinav
"""
#!/usr/bin/python
import numpy as np;
from scipy.misc import toimage;
from scipy.spatial import distance;
import struct;

fp = open('train-images.idx3-ubyte', 'rb');
fp.read(4); #magic no.
size_training_data = struct.unpack('>i', fp.read(4))[0]; # Unpacking as big-endian
print (size_training_data);
row_training_data = struct.unpack('>i', fp.read(4))[0];
print (row_training_data);
col_training_data = struct.unpack('>i', fp.read(4))[0];
print (col_training_data);

#training_data = [];
training_data = np.zeros((size_training_data, row_training_data * col_training_data), dtype=np.int);

for i in range(0,size_training_data):
    for j in range(0, row_training_data * col_training_data):
        training_data[i][j] = struct.unpack('>B', fp.read(1))[0];
        
fp.close(); 




fp = open('t10k-images.idx3-ubyte', 'rb');
fp.read(4); #magic no.
size_test_data = struct.unpack('>i', fp.read(4))[0]; # Unpacking as big-endian
print (size_test_data);
row_test_data = struct.unpack('>i', fp.read(4))[0];
print (row_test_data);
col_test_data = struct.unpack('>i', fp.read(4))[0];
print (col_test_data);

#test_data = [];
test_data = np.zeros((size_test_data, row_test_data * col_test_data), dtype=np.int);

for i in range(0,size_test_data):
    for j in range(0, row_test_data * col_test_data):
        test_data[i][j] = struct.unpack('>B', fp.read(1))[0];
        
fp.close(); 




fp = open('train-labels.idx1-ubyte', 'rb');
fp.read(4); #magic no.
size_training_label = struct.unpack('>i', fp.read(4))[0]; # Unpacking as big-endian
print (size_training_label);

#training_label = [];
training_label = np.zeros(size_training_label, dtype=np.int);

for i in range(0,size_training_label):
    training_label[i] = struct.unpack('>B', fp.read(1))[0];
        
fp.close(); 





fp = open('t10k-labels.idx1-ubyte', 'rb');
fp.read(4); #magic no.
size_test_label = struct.unpack('>i', fp.read(4))[0]; # Unpacking as big-endian
print (size_test_label);

#test_label = [];
test_label = np.zeros(size_test_label, dtype=np.int);

for i in range(0,size_test_label):
    test_label[i] = struct.unpack('>B', fp.read(1))[0];
        
fp.close(); 

#toimage(np.reshape(training_data[0], (28,28))).show()

'''
predicted_label = np.zeros(size_test_data, dtype=np.int);

for i in range(0,size_test_data): 
    min_i = 0;
    temp = np.tile(test_data[i], (size_training_data, 1));
    temp = (training_data - temp)**2;  
    temp_sum = np.sum(temp,1);
    min_i = np.argmin(temp_sum);
    predicted_label[i] = training_label[min_i];
    
hit=0;

for i in range(0,size_test_data): 
    if predicted_label[i] == test_label[i]:
        hit = hit+1
        
print (float(hit/size_test_data)) 
'''

data0 = []
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []
data8 = []
data9 = []

k=0;

for i in range(0,size_training_data):
    if training_label[i] == 0:
        data0.append(training_data[i]);
    if training_label[i] == 1:
        data1.append(training_data[i]);
    if training_label[i] == 2:
        data2.append(training_data[i]);
    if training_label[i] == 3:
        data3.append(training_data[i]);
    if training_label[i] == 4:
        data4.append(training_data[i]);
    if training_label[i] == 5:
        data5.append(training_data[i]);
    if training_label[i] == 6:
        data6.append(training_data[i]);
    if training_label[i] == 7:
        data7.append(training_data[i]);
    if training_label[i] == 8:
        data8.append(training_data[i]);
    if training_label[i] == 9:
        data9.append(training_data[i]);
        
imgsize = row_training_data * col_training_data

tdata0 = np.reshape(data0, (len(data0), imgsize))
tdata1 = np.reshape(data1, (len(data1), imgsize))
tdata2 = np.reshape(data2, (len(data2), imgsize))
tdata3 = np.reshape(data3, (len(data3), imgsize))
tdata4 = np.reshape(data4, (len(data4), imgsize))
tdata5 = np.reshape(data5, (len(data5), imgsize))
tdata6 = np.reshape(data6, (len(data6), imgsize))
tdata7 = np.reshape(data7, (len(data7), imgsize))
tdata8 = np.reshape(data8, (len(data8), imgsize))
tdata9 = np.reshape(data9, (len(data9), imgsize))

avg0 = np.sum(tdata0,0)
avg1 = np.sum(tdata1,0)
avg2 = np.sum(tdata2,0)
avg3 = np.sum(tdata3,0)
avg4 = np.sum(tdata4,0)
avg5 = np.sum(tdata5,0)
avg6 = np.sum(tdata6,0)
avg7 = np.sum(tdata7,0)
avg8 = np.sum(tdata8,0)
avg9 = np.sum(tdata9,0)

size_new = 10000
new_data = np.zeros((size_new, row_training_data * col_training_data), dtype=np.int);
k=0;

def func(tdata, avg):
    global k
    global new_data
    global size_new
    (r,c) = np.shape(tdata)
    temp = np.tile(avg, (r, 1));
    temp = (tdata - temp)**2;  
    temp_sum = np.sum(temp,1);
    max_i = np.argmax(temp_sum);
    min_i = 0
    
    for i in range(0,int(size_new/10)):    
        min_i = np.argmin(temp_sum);        
        new_data[k] = tdata[min_i];
        tdata[min_i] = tdata[max_i];
        k=k+1;
        

func(tdata0, avg0)
func(tdata1, avg1)
func(tdata2, avg2)
func(tdata3, avg3)
func(tdata4, avg4)
func(tdata5, avg5)
func(tdata6, avg6)
func(tdata7, avg7)
func(tdata8, avg8)
func(tdata9, avg9)



predicted_label = np.zeros(size_test_data, dtype=np.int);

for i in range(0,size_test_data): 
    min_i = 0;
    temp = np.tile(test_data[i], (size_new, 1));
    temp = (new_data - temp)**2;  
    temp_sum = np.sum(temp,1);
    min_i = np.argmin(temp_sum);
    predicted_label[i] = int(min_i/(int(size_new/10)));

hit = (predicted_label == test_label)
        
print (float(np.sum(hit)/size_test_data))
    




    




