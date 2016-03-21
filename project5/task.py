# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 23:09:43 2016

@author: Abhinav
"""

import numpy as np
import os

os.chdir('F:\\UCSD\\ML\\5')

#%%

f = open('data1.txt','r')
data = f.read()
f.close()

data = data.split()
x = []
y = []

mod = 3

for i in range(len(data)):
    
    if (i+1)%mod == 0:
        
        y.append(int(data[i]))
        x.append([int(j) for j in data[i-mod+1:i]] + [1])

#%%

p = len(x[0])
n = len(x)

data = np.zeros((n,p+1), dtype=int)
x = np.reshape(x,(n,p))
y = np.reshape(y,(n))

for i in range(n):
    data[i] = list(x[i])+[y[i]]
    

#%%

T=10
l = 1

#c = np.zeros(T, dtype=int)
#w = np.zeros((T,p))
w = np.zeros(p)

c = 1
cl = []
wl = []

while T:
    
    data = np.random.permutation(data)
    
    for i in range(n):
        
        y[i] = data[i][p]
        x[i] = data[i][0:p]
        
        if int(y[i]*(np.inner(w,x[i]))) <= 0:
            
            wl.append(w)
            w = w + y[i]*x[i]
            cl.append(c)
            c = 1
            #w[l+1] = w[l] + y[i]*x[i]
            #c[l+1] = 1
            l = l + 1
            
        else:
            
            #c[l] = c[l] + 1
            c = c+1
            
    T = T-1
 
index = np.argmax(cl)
wt = wl[index]    

#%%
import matplotlib.pyplot as plt

plt.axis([0, 15, 0, 15])
t = np.arange(0, 100, 0.5)
plt.plot([x[i][0] for i in range(n) if y[i]<0], [x[i][1] for i in range(n) if y[i]<0], 'ro')
plt.plot([x[i][0] for i in range(n) if y[i]>0], [x[i][1] for i in range(n) if y[i]>0], 'bo')
plt.plot(t, (-1*wt[2] - 1*wt[0]*t)/wt[1],'g--')

plt.show()

#%%
    
def decision_boundary(cll, wll, ppp):
    Z = np.zeros(len(ppp))
    
    for i in range(len(ppp)):
        Z[i] = np.sign(np.sum([cll[j] * np.sign(np.inner(wll[j], ppp[i])) for j in range(len(wll))]))
            
    return Z
    
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

#X = np.array([[2,1],[3,4],[4,2],[3,1]])
#Y = np.array([0,0,1,1])
h = .1  # step size in the mesh


X = x
Y = y
print(len(X))

# we create an instance of SVM and fit our data. We do not scale our
# data since we want to plot the support vectors


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
fig, ax = plt.subplots()
pp = np.c_[xx.ravel(), yy.ravel()]
print(len(pp[0]))
print(len(pp))

ppp = np.zeros((len(pp),p))
wll = np.zeros((len(wl),p))
cll = np.zeros(len(cl), dtype=int)

for i in range(len(pp)):
    ppp[i] = list(pp[i])+[1.0]
    
for i in range(len(wl)):
    wll[i] = wl[i]

for i in range(len(cl)):
    cll[i] = cl[i]   
    
Z = decision_boundary(cll, wll, ppp)

#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
print(Z.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
ax.axis('off')

# Plot also the training points
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

ax.set_title('Voted Perceptron')
plt.show()


        
        
        
        
        
        
        