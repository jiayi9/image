
#############################################################

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

fit = cv2.findContours(X,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

fit[0]

fit[1]
fit[1][0]
fit[1][0][0]

points = fit[1][0]

# X is changed as well
# image = cv2.circle(X.copy(), (1,1), 0, 255, -1) 

image= X.copy()
for point in points:
    x = point[0][0]
    y = point[0][1]
    print(x,y)
    cv2.circle(image, (x,y), 0, 255, -1) 

plt.imshow(image)

image2 = X.copy()
image2 = cv2.drawContours(image2, fit[1], 0, 255, 0)
plt.imshow(X)
plt.imshow(image2)


fit[2]

################################################################

X = np.random.uniform(0, 255, size = (100,100)).astype(np.uint8)

plt.imshow(X)

fit = cv2.findContours(X,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

fit[1]

image = X.copy()

image = cv2.drawContours(image, fit[1], 0, 255, 0)

plt.imshow(image)
