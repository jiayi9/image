

from matplotlib import pyplot as plt
import numpy as np
import cv2

X = np.array([
        [0,0,0,0,0],
        [0,10,15,15,0],
        [0,10,30,10,0],
        [0,10,10,10,0],
        [0,0,0,0,0]
        ], dtype = np.uint8)


histogram = cv2.calcHist(images = [X], 
    channels = [0], 
    mask = None, 
    histSize = [256], 
    ranges = [0, 256])

plt.imshow(histogram)

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0, 256]) # <- named arguments do not work here
plt.plot(histogram) # <- or here
plt.show()



from PIL import Image

img = np.array(Image.open("C:/LV_CHAO_IMAGE/9sky/train_3/bad_3/0e852267f20f460fba981f0fcaa7f0b7.jpg"))

plt.imshow(img)


histogram = cv2.calcHist(images = [img], 
    channels = [0], 
    mask = None, 
    histSize = [256], 
    ranges = [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim([0, 256]) # <- named arguments do not work here
plt.plot(histogram) # <- or here
plt.show()

retval, threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

plt.imshow(threshold)

cv2.imwrite("C:/daten/delete.jpg", threshold)
