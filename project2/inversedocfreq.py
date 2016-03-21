# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:13:58 2016

@author: Abhinav
"""
#!/usr/bin/python

import os

path = "C:\\Users\\Abhinav\\Downloads\\ML\\"

os.chdir(path)
current  = os.getcwd()
print (current)

l = os.listdir()
bydate = []
for folder in l:
    if folder.startswith('20news'):
        bydate.append(folder)
        
test_labels = []
train_labels = []

for folder in bydate:
    if folder.find('test'):
        test_labels = os.listdir(path+folder)
    if folder.find('train'):
        train_labels = os.listdir(path+folder)
        os.chdir(path+folder)
    else:
        continue

train_files = []
pi = []
current  = os.getcwd()
print (current)

for folder in train_labels:
    train_files.append(os.listdir(folder))

for l in train_files:
    pi.append(len(l))
    
filecount = sum(pi)
    
for l in pi:
    pi[pi.index(l)] = l/filecount
    
print (sum(pi))

i=0
data = []
delimiters = [':', ';', '"', '\n', ',', '<', '>', '(', ')', '[', ']', '. ', '\'', '?']

for l in train_files:    
    os.chdir(current+'\\'+train_labels[i]+'\\')
    i = i + 1
    var = ''
    
    for file in l:
        fd = open(file, 'r')
        var = var + fd.read()
        fd.close()
        
    for file in delimiters:
        var = var.replace(file, ' ')
    
    data.append(var.split())


