# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 00:39:41 2016

@author: Abhinav
"""



#%%
import os
os.chdir('C:\\Users\\Abhinav\\Downloads\\ML\\20news-bydate\\matlab\\')
f = open('train.map', 'r')
var = 'init'
train_labels = []
train_label_val = []
temp = []

while var != '':
    var = f.readline()
    
    if var != '':
        temp.append(var.split())
    
f.close()

for l in temp:
    train_labels.append(l[0])
    train_label_val.append(l[1])
    
    
    
    
    
    
    

f = open('train.label', 'r')
var = 'init'
temp = []

while var != '':
    var = f.readline()
    
    if var != '':
        temp.append(var.split())
        
f.close()

train_labels_count = [0] * len(train_labels)
Pi = [0] * len(train_labels)

for l in temp:
    train_labels_count[int(l[0])-1] = train_labels_count[int(l[0])-1] + 1

total_labels = sum(train_labels_count)


for l in range(20):
    Pi[l] = train_labels_count[l]/total_labels










f = open('train.data', 'r')
var = 'init'
temp = []

freq = []

while var != '':
    var = f.readline()
    
    if var != '':
        temp.append(var.split())

classwords = []
classwordcount = []
words = []
count = []
counter = 0
var = train_labels_count[counter]
i = 0



for l in temp:
    
    if int(l[0]) <= var:
        
        if int(l[1]) not in words:   
            words.append(int(l[1])) 
            count.append(int(l[2])) 
        else:
            i = words.index(int(l[1]))
            count[i] = count[i] + int(l[2])         
            
    else:
        
        classwords.append(words)
        classwordcount.append(count)
        count = []
        words = []
        words.append(int(l[1])) 
        count.append(int(l[2]))             
        counter = counter + 1
        var = var + train_labels_count[counter]

classwords.append(words)
classwordcount.append(count)

f.close()

#%%

import numpy as np

arr = np.ones((20, 61188), dtype=np.int)

for l in classwords:
    for i in l:
        var = l[l.index(i)]
        arr[classwords.index(l)][var-1] = arr[classwords.index(l)][var-1] + classwordcount[classwords.index(l)][l.index(i)]


prob = np.zeros((20, 61188))
counter = 0

for l in arr:
    var = sum(l)
    prob[counter] = [i/var for i in l]
    counter = counter + 1  
    
#%%   
    
f = open('test.label', 'r')
var = 'init'
temp = []

while var != '':
    var = f.readline()
    
    if var != '':
        temp.append(var.split())


f.close()

test_labels = []

for l in temp:
    test_labels.append(int(l[0]))



f = open('test.data', 'r')
var = 'init'
temp = []

while var != '':
    var = f.readline()
    
    if var != '':
        temp.append(var.split())


f.close()

classwords = []
classwordcount = []
words = []
count = []
var = 1
i = 0



for l in temp:
    
    if int(l[0]) == var:         
        words.append(int(l[1])) 
        count.append(int(l[2]))       
    
    else:
        var = int(l[0])
        classwords.append(words)
        classwordcount.append(count)
        count = []
        words = []
        words.append(int(l[1])) 
        count.append(int(l[2]))


classwords.append(words)
classwordcount.append(count)
   
    
    
    
#%%    
    
    
    
    
    
    
    





import math
var = 0
temp = np.zeros(20, dtype=np.int)
labels = []
i=0


test = np.zeros((7505,61188), dtype=np.int)
counter = -1

for l in classwords:
    counter = counter + 1
    for i in range(len(l)):
        test[counter][l[i]-1] = classwordcount[classwords.index(l)][i]
        
#%%
        

pii = np.log(Pi)
probi = np.log(prob)
temp = np.zeros(20)
labels = []
test = test + 1
test = np.log(test)

for l in test:
    ##l = [np.log(1+i) for i in l]
    for count in range(20):
        temp[count] = pii[count] + np.dot(l, probi[count])
        count = count + 1
    i = np.argmax(temp)
    labels.append(i+1)
    
hit = (np.array(labels) == np.array(test_labels))
hit = np.array(hit)
print (np.sum(hit))
    
        

    

