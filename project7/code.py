# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 21:36:34 2016

@author: Abhinav
"""

f = open('cities.txt', 'r')
var = f.readline()
cities = []

while var:
    cities.append(var.split('\n')[0])
    var = f.readline()
    
f.close()

f = open('distances.txt', 'r')
var = f.readline()
distances = []

while var:
    l = var.split('\n')[0].split(',')
    l = [int(i) for i in l]
    distances.append(l)
    var = f.readline()
    
f.close()

#%%
import numpy as np

k=2
n = len(cities)

D = np.reshape(distances, (n,n))
D = np.square(D)
I = np.identity(n)
m = np.ones((n,n))
H = I - (1/n * m)

B = -1/2 * (H * D * H)

from scipy import linalg

[u, s, v] = linalg.svd(B)

#%%
 

Y = u * np.sqrt(s)
Yk = Y[:,:2]

import matplotlib.pyplot as plt

#plt.axis([-3, 3, 2, 3])
t = np.arange(0, 100, 0.5)
plt.plot([Yk[i][0] for i in range(n)], [Yk[i][1] for i in range(n)], 'ro')

plt.show()

#%%

import sklearn.manifold

model = sklearn.manifold.MDS()
model.fit(Yk)

