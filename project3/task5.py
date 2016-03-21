# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:43:09 2016

@author: Abhinav
"""
#!/usr/bin/python

import numpy as np
from scipy.misc import toimage
from scipy.stats import multivariate_normal
from scipy.spatial import distance
import struct
import os
import pickle
import random

os.chdir('C:\\Users\\Abhinav\\Downloads\\ML\\3')
print (os.getcwd())

#%%

trdata = pickle.load(open('training_data', 'rb'))
trlabel = pickle.load(open('training_label', 'rb'))
tsdata = pickle.load(open('test_data', 'rb'))
tslabel = pickle.load(open('test_label', 'rb'))

size_trdata = np.shape(trdata)[0]
size_trlabel = np.shape(trlabel)[0]
size_tsdata = np.shape(tsdata)[0]
size_tslabel = np.shape(tslabel)[0]
rows_trdata = 28
cols_trdata = 28
rows_tsdata = 28
cols_tsdata = 28

#%%

fp = open('train-images.idx3-ubyte', 'rb');
fp.read(4); #magic no.
size_trdata = struct.unpack('>i', fp.read(4))[0]; # Unpacking as big-endian
rows_trdata = struct.unpack('>i', fp.read(4))[0];
cols_trdata = struct.unpack('>i', fp.read(4))[0];

trdata = np.zeros((size_trdata, rows_trdata * cols_trdata), dtype=np.int);

for i in range(0,size_trdata):
    for j in range(0, rows_trdata * cols_trdata):
        trdata[i][j] = struct.unpack('>B', fp.read(1))[0];
        
fp.close()

fp = open('train-labels.idx1-ubyte', 'rb');
fp.read(4); #magic no.
size_trlabel = struct.unpack('>i', fp.read(4))[0]; # Unpacking as big-endian

trlabel = np.zeros(size_trlabel, dtype=np.int)

for i in range(0,size_trlabel):
    trlabel[i] = struct.unpack('>B', fp.read(1))[0]
        
fp.close()
pickle.dump(trdata, open('training_data', 'wb'))
pickle.dump(trlabel, open('training_label', 'wb'))

#%%

fp = open('t10k-images.idx3-ubyte', 'rb')
fp.read(4) #magic no.
size_tsdata = struct.unpack('>i', fp.read(4))[0] # Unpacking as big-endian
rows_tsdata = struct.unpack('>i', fp.read(4))[0]
cols_tsdata = struct.unpack('>i', fp.read(4))[0]

tsdata = np.zeros((size_tsdata, rows_tsdata * cols_tsdata), dtype=np.int)

for i in range(0,size_tsdata):
    for j in range(0, rows_tsdata * cols_tsdata):
        tsdata[i][j] = struct.unpack('>B', fp.read(1))[0]
        
fp.close()

fp = open('t10k-labels.idx1-ubyte', 'rb');
fp.read(4) #magic no.
size_tslabel = struct.unpack('>i', fp.read(4))[0] # Unpacking as big-endian

tslabel = np.zeros(size_tslabel, dtype=np.int)

for i in range(0,size_tslabel):
    tslabel[i] = struct.unpack('>B', fp.read(1))[0]
        
fp.close()

pickle.dump(tsdata, open('test_data', 'wb'))
pickle.dump(tslabel, open('test_label', 'wb'))

#%%

toimage(np.reshape(trdata[1030], (28,28))).show()

#%%

target = 10000
valid_trdata = np.zeros((target, rows_trdata * cols_trdata), dtype=np.int);
valid_trlabel = np.zeros(target, dtype=np.int);

num = np.random.randint(size_trdata,size=target)

temptrdata = trdata
temptrlabel = trlabel

for i in range(0,target):
    valid_trdata[i] = trdata[num[i]]      
    valid_trlabel[i] = trlabel[num[i]]
    temptrdata = np.delete(trdata,num[i],0)
    temptrlabel = np.delete(trlabel,num[i],0)
    
size_trdata = np.shape(trdata)[0]
size_trlabel = np.shape(trlabel)[0]

#%%
valid = random.sample(range(size_trdata),target)
validation_label = [trlabel[i] for i in valid]
validation_set = [trdata[i] for i in valid]

newtrlabel = []
newtrdata = []

for i in range(size_trdata):
    if i not in valid:
        newtrlabel.append(trlabel[i])
        newtrdata.append(trdata[i])

#%%
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

for i in range(0,size_trdata):
    if trlabel[i] == 0:
        data0.append(trdata[i]);
    if trlabel[i] == 1:
        data1.append(trdata[i]);
    if trlabel[i] == 2:
        data2.append(trdata[i]);
    if trlabel[i] == 3:
        data3.append(trdata[i]);
    if trlabel[i] == 4:
        data4.append(trdata[i]);
    if trlabel[i] == 5:
        data5.append(trdata[i]);
    if trlabel[i] == 6:
        data6.append(trdata[i]);
    if trlabel[i] == 7:
        data7.append(trdata[i]);
    if trlabel[i] == 8:
        data8.append(trdata[i]);
    if trlabel[i] == 9:
        data9.append(trdata[i]);

tdata0 = np.reshape(data0, (len(data0), rows_trdata * cols_trdata))
tdata1 = np.reshape(data1, (len(data1), rows_trdata * cols_trdata))
tdata2 = np.reshape(data2, (len(data2), rows_trdata * cols_trdata))
tdata3 = np.reshape(data3, (len(data3), rows_trdata * cols_trdata))
tdata4 = np.reshape(data4, (len(data4), rows_trdata * cols_trdata))
tdata5 = np.reshape(data5, (len(data5), rows_trdata * cols_trdata))
tdata6 = np.reshape(data6, (len(data6), rows_trdata * cols_trdata))
tdata7 = np.reshape(data7, (len(data7), rows_trdata * cols_trdata))
tdata8 = np.reshape(data8, (len(data8), rows_trdata * cols_trdata))
tdata9 = np.reshape(data9, (len(data9), rows_trdata * cols_trdata))

mean0 = np.mean(tdata0, axis=0)
mean1 = np.mean(tdata1, axis=0)
mean2 = np.mean(tdata2, axis=0)
mean3 = np.mean(tdata3, axis=0)
mean4 = np.mean(tdata4, axis=0)
mean5 = np.mean(tdata5, axis=0)
mean6 = np.mean(tdata6, axis=0)
mean7 = np.mean(tdata7, axis=0)
mean8 = np.mean(tdata8, axis=0)
mean9 = np.mean(tdata9, axis=0)

#%%

cov0 = np.cov(np.transpose(tdata0))
cov1 = np.cov(np.transpose(tdata1))
cov2 = np.cov(np.transpose(tdata2))
cov3 = np.cov(np.transpose(tdata3))
cov4 = np.cov(np.transpose(tdata4))
cov5 = np.cov(np.transpose(tdata5))
cov6 = np.cov(np.transpose(tdata6))
cov7 = np.cov(np.transpose(tdata7))
cov8 = np.cov(np.transpose(tdata8))
cov9 = np.cov(np.transpose(tdata9))

#%%
y = np.zeros((10,10000))
labels = []

total = np.shape(data0)[0] + np.shape(data1)[0] + np.shape(data2)[0] + np.shape(data3)[0] + np.shape(data4)[0] + np.shape(data5)[0] + np.shape(data6)[0] + np.shape(data7)[0] + np.shape(data8)[0] + np.shape(data9)[0]

theta = 2800

y[0] = multivariate_normal.logpdf(tsdata, mean0, cov0 + theta*np.eye(784,784)) + np.log(np.shape(data0)[0]/total)
y[1] = multivariate_normal.logpdf(tsdata, mean1, cov1 + theta*np.eye(784,784)) + np.log(np.shape(data1)[0]/total)
y[2] = multivariate_normal.logpdf(tsdata, mean2, cov2 + theta*np.eye(784,784)) + np.log(np.shape(data2)[0]/total)
y[3] = multivariate_normal.logpdf(tsdata, mean3, cov3 + theta*np.eye(784,784)) + np.log(np.shape(data3)[0]/total)
y[4] = multivariate_normal.logpdf(tsdata, mean4, cov4 + theta*np.eye(784,784)) + np.log(np.shape(data4)[0]/total)
y[5] = multivariate_normal.logpdf(tsdata, mean5, cov5 + theta*np.eye(784,784)) + np.log(np.shape(data5)[0]/total)
y[6] = multivariate_normal.logpdf(tsdata, mean6, cov6 + theta*np.eye(784,784)) + np.log(np.shape(data6)[0]/total)
y[7] = multivariate_normal.logpdf(tsdata, mean7, cov7 + theta*np.eye(784,784)) + np.log(np.shape(data7)[0]/total)
y[8] = multivariate_normal.logpdf(tsdata, mean8, cov8 + theta*np.eye(784,784)) + np.log(np.shape(data8)[0]/total)
y[9] = multivariate_normal.logpdf(tsdata, mean9, cov9 + theta*np.eye(784,784)) + np.log(np.shape(data9)[0]/total)

y = np.transpose(y)

for i in range(0,10000):
    labels.append(np.argmax(y[i]))
    
hits = (tslabel == labels)
print (np.sum(hits)/size_tsdata)

#%%
y = np.zeros((10,target))
labels = []

total = np.shape(data0)[0] + np.shape(data1)[0] + np.shape(data2)[0] + np.shape(data3)[0] + np.shape(data4)[0] + np.shape(data5)[0] + np.shape(data6)[0] + np.shape(data7)[0] + np.shape(data8)[0] + np.shape(data9)[0]

theta = 2800

y[0] = multivariate_normal.logpdf(tsdata, mean0, cov0 + theta*np.eye(784,784)) + np.log(np.shape(data0)[0]/total)
y[1] = multivariate_normal.logpdf(tsdata, mean1, cov1 + theta*np.eye(784,784)) + np.log(np.shape(data1)[0]/total)
y[2] = multivariate_normal.logpdf(tsdata, mean2, cov2 + theta*np.eye(784,784)) + np.log(np.shape(data2)[0]/total)
y[3] = multivariate_normal.logpdf(tsdata, mean3, cov3 + theta*np.eye(784,784)) + np.log(np.shape(data3)[0]/total)
y[4] = multivariate_normal.logpdf(tsdata, mean4, cov4 + theta*np.eye(784,784)) + np.log(np.shape(data4)[0]/total)
y[5] = multivariate_normal.logpdf(tsdata, mean5, cov5 + theta*np.eye(784,784)) + np.log(np.shape(data5)[0]/total)
y[6] = multivariate_normal.logpdf(tsdata, mean6, cov6 + theta*np.eye(784,784)) + np.log(np.shape(data6)[0]/total)
y[7] = multivariate_normal.logpdf(tsdata, mean7, cov7 + theta*np.eye(784,784)) + np.log(np.shape(data7)[0]/total)
y[8] = multivariate_normal.logpdf(tsdata, mean8, cov8 + theta*np.eye(784,784)) + np.log(np.shape(data8)[0]/total)
y[9] = multivariate_normal.logpdf(tsdata, mean9, cov9 + theta*np.eye(784,784)) + np.log(np.shape(data9)[0]/total)

y = np.transpose(y)
dis = np.zeros(size_tsdata)
j1 = 0
j2 = 0

for i in range(0,size_tsdata):
    j1 = np.argmax(y[i])
    j2 = np.argmin(y[i])
    y[i][j1] = y[i][j2]
    j2 = np.argmax(y[i])
    dis[i] = (y[i][j1] - y[i][j2])/y[i][j1]
    labels.append(j1)
    
f = 0.05
abstain = int(f * size_tsdata)
    
for i in range(abstain):
    j1 = np.argmin(dis)
    labels[j1] = -1
    dis[j1] = np.argmax(dis)
    
hits = (tslabel == labels)
print (np.sum(hits)/(size_tsdata-abstain))




#%%
miss = []

for i in range(np.shape(hits)[0]):
    if hits[i] == False:
        miss.append(i)
        
num = np.random.randint(len(miss),size=5)
z = np.zeros((5,10))

#%%
from scipy.misc import imsave

for i in range(5):
    #toimage(np.reshape(tsdata[miss[i]], (28,28))).show()
    imsave(str(i)+'_5.jpg', np.reshape(tsdata[miss[i]], (28,28)), format=None)
    
    z[i][0] = multivariate_normal.logpdf(tsdata[miss[i]], mean0, cov0 + theta*np.eye(784,784))
    z[i][1] = multivariate_normal.logpdf(tsdata[miss[i]], mean1, cov1 + theta*np.eye(784,784))
    z[i][2] = multivariate_normal.logpdf(tsdata[miss[i]], mean2, cov2 + theta*np.eye(784,784))
    z[i][3] = multivariate_normal.logpdf(tsdata[miss[i]], mean3, cov3 + theta*np.eye(784,784))
    z[i][4] = multivariate_normal.logpdf(tsdata[miss[i]], mean4, cov4 + theta*np.eye(784,784))
    z[i][5] = multivariate_normal.logpdf(tsdata[miss[i]], mean5, cov5 + theta*np.eye(784,784))
    z[i][6] = multivariate_normal.logpdf(tsdata[miss[i]], mean6, cov6 + theta*np.eye(784,784))
    z[i][7] = multivariate_normal.logpdf(tsdata[miss[i]], mean7, cov7 + theta*np.eye(784,784))
    z[i][8] = multivariate_normal.logpdf(tsdata[miss[i]], mean8, cov8 + theta*np.eye(784,784))
    z[i][9] = multivariate_normal.logpdf(tsdata[miss[i]], mean9, cov9 + theta*np.eye(784,784))
    
    
pickle.dump(z, open('z', 'wb'))
        
        
#%%
import matplotlib.pyplot as plt
lines = plt.plot(-4, 0, 0, 3)
        

        
        


    




    




