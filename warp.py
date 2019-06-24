# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:40:52 2019

@author: LUI8WX
"""

import numpy as np
import cv2

from skimage import io

img = cv2.imread('C:/Suphina/NOK_1.BMP') 

rows,cols,_ = img.shape 

io.imshow(img)
#cv2.imshow("OpenCV",img)

points1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
points2 = np.float32([[0,0],[300,0],[0,300],[300,300]])


points1 = np.float32([[150,150],[500,150],[150,400],[500,400]]) 


points1 = np.float32([[150,150],[180,150],[150,180],[180,180]]) 
(cols, rows)


img = cv2.imread('C:/Suphina/NOK_1.BMP') 

points1 = np.float32([[10,10],[586,10],[603,10],[586,603]]) 
points2 = np.float32([[100,100],[300,100],[300,100],[300,300]])
matrix = cv2.getPerspectiveTransform(points1,points2)

cv2.circle(img, tuple(points1[0]),10,(255,0,0),-1)
cv2.circle(img, tuple(points1[1]),10,(255,0,0),-1)
cv2.circle(img, tuple(points1[2]),10,(255,0,0),-1)
cv2.circle(img, tuple(points1[3]),10,(255,0,0),-1)

output = cv2.warpPerspective(img, matrix, (cols, rows))
io.imshow(img)
io.imshow(output)




img = cv2.imread('C:/Suphina/NOK_1.BMP') 

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])

(cols, rows)

pts1 = np.float32([[0,0],[586,0],[0,603]])
pts2 = np.float32([[100,100],[200,50],[100,250]])

#pts2 = pts1

pts1 = np.float32([[0,0],[586,0],[0,603]])
pts2 = np.float32([[100,100],[200,50],[100,250]])



cv2.circle(img, tuple(pts1[0]),10,(0,255,255),-1)
cv2.circle(img, tuple(pts1[1]),10,(0,255,255),-1)
cv2.circle(img, tuple(pts1[2]),10,(0,255,255),-1)
#cv2.circle(img, tuple(pts2[0]),10,(255,0,0),-1)
#cv2.circle(img, tuple(pts2[1]),10,(255,0,0),-1)
#cv2.circle(img, tuple(pts2[2]),10,(255,0,0),-1)


M_affine = cv2.getAffineTransform(pts1,pts2)
img_affine = cv2.warpAffine(img, M_affine, (cols, rows))
io.imshow(img)
io.imshow(img_affine)










