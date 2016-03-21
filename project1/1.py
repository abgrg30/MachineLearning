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
import random;

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

target = 10000
short_training_data = np.zeros((target, row_training_data * col_training_data), dtype=np.int);
short_training_label = np.zeros(target, dtype=np.int);

num = random.randint(0,size_training_data-1)

short_training_data[0] = training_data[num];
short_training_label[0] = training_label[num];

i=1

while i < target:
   num = random.randint(0,size_training_data-1)
   
   temp = np.tile(training_data[num], (i, 1));
   temp = (short_training_data[0:i] - temp)**2;  
   temp_sum = np.sum(temp,1);
   min_i = np.argmin(temp_sum);
   label = short_training_label[min_i];
   
   if label != training_label[num]:
       short_training_data[i] = training_data[num]
       short_training_label[i] = training_label[num]
       i = i + 1
       
       
predicted_label = np.zeros(size_test_data, dtype=np.int);

for i in range(0,size_test_data): 
    min_i = 0;
    temp = np.tile(test_data[i], (target, 1));
    temp = (short_training_data - temp)**2;  
    temp_sum = np.sum(temp,1);
    min_i = np.argmin(temp_sum);
    predicted_label[i] = short_training_label[min_i];
    
hit=0;

hit = (predicted_label == test_label)
        
print (float(np.sum(hit)/size_test_data))
       
       
   
   
   
   
   
   
   
   