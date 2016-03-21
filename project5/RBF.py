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
import math
sigma = 5
alpha = np.zeros(n)
flag = 1

while 1:    
    
    for i in range(n):
        
        wx = int(np.sum([alpha[j] * y[j] * math.exp(-1/2 * np.square(np.linalg.norm(x[j] - x[i])/sigma)) for j in range(n)]))
        
        if (y[i] * wx) <= 0:

            alpha[i] = alpha[i] + 1
            flag = 0
            
    if flag == 1:
        break
    else:
        flag = 1
    
root = np.sqrt(2)

#%%


def decision_boundaryk(cll, wll, ppp):
    Z = np.zeros(len(ppp))
    
    wn = np.sum(cll[j]*wll[j] for j in range(l-1))
    Z = np.inner(wn, ppp)
    Z = np.sign(Z)
    
    return Z
    
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

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

for i in range(len(pp)):
    ppp[i] = list(pp[i])+[1.0]   
    
#Z = decision_boundaryavg(cll, wll, ppp)

Z = np.zeros(len(ppp))
for i in range(len(pp)):
    Z[i] = np.sign(np.sum(alpha[j] * y[j] * math.exp(-1/2 * np.square(np.linalg.norm(x[j] - ppp[i])/sigma)) for j in range(n)))


#Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
print(Z.shape)
ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
ax.axis('off')

# Plot also the training points
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

ax.set_title('RBF Perceptron')
plt.show()


   
        
        
        
        
        
        
        
        