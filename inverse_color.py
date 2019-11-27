# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 13:42:56 2019

@author: LUI8WX
"""




from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


PATH = "//bosch.com/dfsrb/DfsCN/loc/Wx/Dept/TEF/60_MFE_Manufacturing_Engineering/06_Data_Analytics/01_Project/MOE/MOE9/atmo3/TEST11_OK.bmp"
x = np.array(Image.open(PATH).convert("L"))

plt.imshow(x)

x2 = 255 - x


plt.imshow(x2)


import cv2

cv2.imwrite("C:/daten/delete.jpg", x2)

ret,thresh1 = cv2.threshold(x2,200,255,cv2.THRESH_BINARY)

plt.imshow(thresh1)

cv2.imwrite("C:/daten/delete.jpg", thresh1)
