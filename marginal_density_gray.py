import cv2
import numpy as np
from matplotlib import pyplot as plt



from PIL import Image

img = np.array(Image.open("N:/RBCD_Data_mining/Data_Analytics_Community/Python/to_CR/bad/4_21_0.53.png"))

#plt.imshow(img)


w, h = img.shape[0:2]

# Threshold to prevent JPG artifacts
#_, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

# Sum pixels along x and y axis
xMean = np.mean(img, axis=0)
yMean = np.mean(img, axis=1)

# Visualize curves
plt.plot(xMean)
plt.plot(yMean)
plt.legend(['x density','y density'], loc='upper left')
plt.show()




# Set up thresholds
xThr = 15
yThr = 15

# Find proper row indices
tmp = np.argwhere(xMean > xThr)
tmp = tmp[np.where((tmp > 20) & (tmp < w - 20))]
x1 = tmp[0]
x2 = tmp[-1]

# Find proper column indices
tmp = np.argwhere(yMean > yThr)
tmp = tmp[np.where((tmp > 20) & (tmp < h - 20))]
y1 = tmp[0]
y2 = tmp[-1]

# Visualize result
out = img

print(x1, x2, y1, y2)

x1,x2,y1,y2 = 1000,1300, 500,1300

#y1,y2,x1,x2 = 500,1500, 800,1500


cv2.rectangle(out, (x1, y1), (x2, y2), 255, 4)

plt.imshow(out)

cv2.imwrite("C:/daten/delete.jpg", out)
