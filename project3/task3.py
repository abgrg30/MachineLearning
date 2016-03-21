# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:06:13 2016

@author: Abhinav
"""

import matplotlib.pyplot as plt
import numpy as np

mean = [0, 0]
cov = [[9, 0], [0, 1]]

x, y = np.random.multivariate_normal(mean, cov, 100).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()


mean = [0, 0]
cov = [[1, -0.75], [-0.75, 1]]

x, y = np.random.multivariate_normal(mean, cov, 100).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.figure, plt.show()