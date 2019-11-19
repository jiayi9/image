

import numpy as np
import cv2
from matplotlib import pyplot as plt

x = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]        
], dtype = np.uint8)

#kernel = np.ones((2,3))

blur = cv2.GaussianBlur(x.copy(),(3,3),0)

plt.imshow(x)

plt.imshow(blur)



x = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,200,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]        
], dtype = np.uint8)

blur = cv2.GaussianBlur(x.copy(),(3,3),0)
print(blur)
x.sum()
blur.sum()





x = np.zeros((7,7)).astype(np.uint8)
x[2,2] = 100
x[5,5] = 200
x

blur = cv2.GaussianBlur(x.copy(),(3,3),0)
print(blur)
x.sum()
blur.sum()




x = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,200,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]        
], dtype = np.uint8)

blur0 = cv2.GaussianBlur(x.copy(),(3,3),0)
blur1 = cv2.GaussianBlur(x.copy(),(3,3),1)
blur2 = cv2.GaussianBlur(x.copy(),(3,3),2)
blur3 = cv2.GaussianBlur(x.copy(),(3,3),3)

print(blur0)
print(blur1)
print(blur2)
print(blur3)

cv2.getGaussianKernel(5,5)

cv2.getGaussianKernel(5,5).sum()







x = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,200,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]        
], dtype = np.uint8)

kernel = np.ones((5,5),np.float32)/25

kernel = np.ones((3,3),np.float32)/9

dst = cv2.filter2D(x.copy(), -1, kernel)

print(dst)








mask = np.zeros(x.shape,np.uint8)
mask[2,2] = 2
mask[4,3] = 6
np.nonzero(mask)
np.transpose(np.nonzero(mask))


#cv2.drawContours(mask,[cnt],0,255,-1)
#pixelpoints = np.transpose(np.nonzero(mask))
#pixelpoints = cv2.findNonZero(mask)



