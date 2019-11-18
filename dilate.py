# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:59:29 2019

@author: LUI8WX
"""

import numpy as np
import cv2
x = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]        
], dtype = np.uint8)
kernel = np.ones((2,3)) + 1
cv2.dilate(x, kernel,1 )





import numpy as np
import cv2
x = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]        
], dtype = np.uint8)

kernel = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
        ])

cv2.dilate(x, kernel,1 )




import numpy as np
import cv2
x = np.zeros((7,7))
x[3,3] = 1
kernel = np.ones((3,5))
cv2.dilate(x, kernel,  iterations = 1)

import numpy as np
import cv2
x = np.zeros((7,7))
x[3,3] = 100
kernel = np.ones((3,3))
cv2.dilate(x, kernel,  iterations = 2)



import numpy as np
import cv2
x = np.zeros((7,7))
x[1,1] = 100
x[5,5] = 200
kernel = np.ones((3,3))
cv2.dilate(x, kernel,  iterations = 1)
cv2.dilate(x, kernel,  iterations = 2)
cv2.dilate(x, kernel,  iterations = 3)
cv2.dilate(x, kernel,  iterations = 4)
cv2.dilate(x, kernel,  iterations = 5)




