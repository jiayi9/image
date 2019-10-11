# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:59:28 2019

@author: LUI8WX
"""

import numpy as np
from matplotlib import pyplot as plt

x = np.array([1,2])
y = np.array([3,4])
np.vstack((x,y))

x = np.array([[1,2], [3,4]])
y = np.array([[5,6]])
np.vstack((x,y))

x = np.array([[1,2], [3,4]])
y = np.array([5,6])
np.vstack((x,y))

x = np.array([[1,2], [3,4]])
y = np.array([[5,6], [7,8]])
np.vstack((x,y))

plt.imshow(np.vstack((x,y)))

x = np.array([[[1,2], [3,4]]])
y = np.array([[[5,6], [7,8]]])
np.vstack((x,y))

plt.imshow(np.vstack((x,y))[0])
plt.imshow(np.vstack((x,y))[1], vmin = 0, vmax = 255)



x = np.array([[1,2], [3,4]])
np.expand_dims(x, 0)
y = np.array([[5,6], [7,8]])
np.expand_dims(y,0)

np.vstack((
        np.expand_dims(x, 0),
        np.expand_dims(y,0)
))




x = np.array([
        [
                [1,2], 
                [3,4]
        ],
        [
                [1,2], 
                [3,4]
        ]                
])

plt.imshow(x[0])
