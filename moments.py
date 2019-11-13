# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:28:46 2019

@author: LUI8WX
"""

from matplotlib import pyplot as plt
import numpy as np

X = np.array([
        [0,0,0,0,0],
        [0,10,15,15,0],
        [0,10,30,10,0],
        [0,10,10,10,0],
        [0,0,0,0,0]
        ], dtype = np.uint8)

X

plt.imshow(X, vmin = 0, vmax = 255)


import cv2
 #cv2.findContours(X,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.moments(X)


# mo00
X.sum()
m = X.shape[0]
n = X.shape[1]
mean_x = int((m - 1)/2)
mean_y = int((n - 1)/2)
# mo01

mo01 = 0
for i in range(m):
    for j in range(n):
        mo01 = mo01 + i*X[i,j]
print(mo01)

mo10 = 0
for i in range(m):
    for j in range(n):
        mo10 = mo10 + j*X[i,j]
print(mo10)


mo02 = 0
for i in range(m):
    for j in range(n):
        mo02 = mo02 + i**2*X[i,j]
print(mo02)


#mu11
w = 0
for i in range(m):
    for j in range(n):
#        w = w + (i- mean_x)*(j-mean_y)*X[i,j]
        w = w + (j- 2)*(i-2)*X[i,j]

print(w)


#mu20

cv2.moments(X)
w = 0
for i in range(m):
    for j in range(n):
#        w = w + (i- mean_x)*(j-mean_y)*X[i,j]
        w = w + ((j- 2)**2)*X[i,j] # a little off
        # wrong w = w + ((j- 3)**2)*X[i,j]
    
print(w)


