# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
os.chdir('F:\\UCSD\\ML\\6')

#%%
lanimal = []
nanimal = []

f = open('classes.txt', 'r')
var = f.readline()

while var:
    lanimal.append(var.split()[0])
    nanimal.append(var.split()[1])
    var = f.readline() 

f.close()

#%%
attributes = []

f = open('predicate-matrix-continuous.txt', 'r')
var = f.readline()

while var:
    temp = var.split()
    attributes.append([float(i) for i in temp])
    var = f.readline()
f.close()

#%%
lfeature = []
nfeature = []

f = open('predicates.txt', 'r')
var = f.readline()

while var:
    lfeature.append(var.split()[0])
    nfeature.append(var.split()[1])
    var = f.readline() 

f.close()

#%%
import sklearn.cluster
import numpy as np

k=10
data = np.reshape(attributes, (50,85))

model = sklearn.cluster.KMeans(k)
model.fit(data)
final = model.labels_

clusters = [[] for i in range(k)]

for i in range(len(final)):
    clusters[final[i]].append(nanimal[i])
    
#%%

import scipy.cluster
from pylab import rcParams
rcParams['figure.figsize'] = 5, 10

hierar = scipy.cluster.hierarchy.linkage(data,'average')
diction = scipy.cluster.hierarchy.dendrogram(hierar, 50, None, None, True, 'right', nanimal)

#%%
import numpy
##import sklearn.decomposition
##model = sklearn.decomposition.PCA(2)
##model.fit(data.T)
#pca = model.components_
#pca = pca.T
[u,s,v] = numpy.linalg.svd(data.T)

pca = u[:,0:2].T
pca = np.dot(pca, data.T).T



#%%
import matplotlib.pyplot as plt

plt.axis([100, 400, -250, 150])
t = np.arange(0, 1000, 0.5)
#fig = plt.figure()
#ax = fig.add_subplot(111)

for i in range(50):
    plt.plot(pca[i][0], pca[i][1], 'ro')
    #plt.annotate(nanimal[i], nanimal[i])
    plt.text(pca[i][0], pca[i][1], nanimal[i])
    ##plt.label(nanimal[i])

#plt.grid()
plt.show()







