# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 19:47:27 2016

Some Matrix Decompositions

@author: abhijit
"""

#%%
import numpy as np
import scipy.linalg as slin
import matplotlib.pyplot as plt


def closest_match():
    a = np.array([[1,2,3],[4,5,6]])
    
    u,s,v = slin.svd(a,full_matrices = False)
    
    """
    Best Rank 1 Approximation
    """
    S = np.diag(s)
    wt = np.dot(S,v)
    
    u_red  = u[:,0][:,None]
    wt_red = wt[0:,][0,None]
    
    a_closest = u_red.dot(wt_red)
    return a_closest    



def Gram_mat(ptsCombo):
    dotP = (ptsCombo.T).dot(ptsCombo)
    return dotP    
    
def ParseCity(fname):
    file = open(fname,'r')
    cities = []
    for line in file:
        cities.append(line.strip())
    file.close()
    return cities

def ParseDistance(fname):
    file = open(fname,'r')
    distance = []
    for line in file:
        line = line.strip()
        distance.append(line.split(','))
    return distance


def MultiDimensionScaling(fDist,fCity):
    cities = ParseCity(fCity)
    distances = ParseDistance(fDist)
    
    ### Do MDS here . then move to a function ###
    D = np.array(distances,np.float)
    
    D = D**2;    
    
    
    r,c = D.shape
    One = np.zeros((r))
    for i in range(r):
        One[i]=1
    One = One[:,None]
    OneOneT = One.dot(One.T)
    H = np.eye(10)-(OneOneT/r)
    B1 = H.dot(D)
    B = -(B1.dot(H))/2
    
    
    u,s,w = np.linalg.svd(B)
    vMan = s
    wMan = u
        
    
    """
    v,w = np.linalg.eig(B)
    SortedInd = np.argsort(v)[r-1::-1]
    vMan = v[SortedInd]
    wMan = w.T[SortedInd]
    """
    for i in range(vMan.shape[0]):
        if vMan[i]<0:
            vMan[i]=0
    vMan = np.power(vMan,0.5);
    vMan = np.diag(vMan)
    
    Y = wMan.dot(vMan)
    
    
    """    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(r):
        ax.plot(Y[i,1],Y[i,2],'bo')
        ax.annotate(xy = (Y[i,1],Y[i,2]),xytext = (cities[i]))
    """
    Y = Y[:,0:2]    
    
    
    # Begin Plotting
    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(
        Y[:, 0], Y[:, 1], marker = 'o',
        cmap = plt.get_cmap('Spectral'))
    for label, x, y in zip(cities, Y[:, 0], Y[:, 1]):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (-10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
    plt.savefig('original.jpg')    
    return Y


if __name__=="__main__" :
    
    
    pt1 = np.array([0,1,0])
    pt2 = np.array([0,1,1])
    pt3 = np.array([1,1,0])
    pt4 = np.array([1,1,1])
    
    
    x = np.concatenate((pt1[:,None],pt2[:,None],pt3[:,None],pt4[:,None]),axis=1)
    B = Gram_mat(x)
    
    fDist = 'distances.txt'
    fCity = 'cities.txt'
    y = MultiDimensionScaling(fDist,fCity)
    
    