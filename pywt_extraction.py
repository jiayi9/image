import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pywt
import pywt.data
import glob

PATH = ""

files = glob.glob(PATH + "*.png")

file = files[1]
img = Image.open(file)
im = np.array(img)

# Load image
original = im

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2

cv2.imwrite(PATH + "dwt/LL.jpg", LL)
cv2.imwrite(PATH + "dwt/LH.jpg", LH)
cv2.imwrite(PATH + "dwt/HL.jpg", HL)
cv2.imwrite(PATH + "dwt/HH.jpg", HH)

fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()


LH2 = (LH - LH.min())/(LH.max() - LH.min())

plt.imshow(LH2, cmap = "gray")

LH3 = 1 - LH2
plt.imshow(LH3, cmap = "gray")


LH4 = LH3*255
LH4 = LH4.astype("uint8")
ret, thresh = cv2.threshold(LH4,90,255,cv2.THRESH_BINARY_INV)
cv2.imwrite(PATH + "dwt/output.jpg", thresh)


im_dwt = thresh


im = im_dwt.copy()

kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations = 1)
cv2.imwrite(PATH + "dwt/opening_1.jpg",opening)


#kernel = np.ones((15,15),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#cv2.imwrite(PATH + "opening_2.jpg",opening)

#kernel = np.ones((15,15),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
#cv2.imwrite(PATH + "opening_2.jpg",opening)


#kernel = np.ones((15,15),np.uint8)
#closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 1)
#cv2.imwrite(PATH + "closing.jpg",closing)

dilate_height = 5
dilate_width = 30
kernel = np.ones((dilate_height,dilate_width),np.uint8)
dilated = cv2.dilate(opening, kernel)
cv2.imwrite(PATH + "dwt/dilate.jpg",dilated)




drawing = im.copy()
opening_copy = opening.copy()
_,contours,hierarchy = cv2.findContours(dilated, 1, 2)
L = []
for index, contour in enumerate(contours):
    print(index)
    
    x,y,w,h = cv2.boundingRect(contour)
    drawing = cv2.rectangle(drawing,(x,y),(x+w,y+h),(255,255,0),3)
    tmp = drawing[y:y+h, x:x+w]
    L.append(tmp)
    cv2.imwrite(PATH + str(index) + ".jpg", tmp)
#    print(cv2.Laplacian(tmp, cv2.CV_64F).var())#tmp.std())



cv2.imwrite(PATH + "drawing.jpg",drawing)    




