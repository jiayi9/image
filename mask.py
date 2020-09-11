
import numpy as np
from matplotlib import pyplot as plt

x = np.random.rand(500, 500)

plt.imshow(x)

x2 = x.copy()


LEFT = np.zeros((600, 300))

RIGHT = np.ones((600, 300))

MERGE = np.concatenate((LEFT, RIGHT),   axis = 1)

plt.imshow(MERGE)

plt.imshow(MERGE.astype(int))

plt.imshow(MERGE.astype(np.int0))

plt.imshow(MERGE.astype(np.int16))


plt.imshow(MERGE, vmax= 255)

plt.imshow(MERGE, cmap = 'gray')

plt.imshow(MERGE, cmap = 'gray', vmax = 255)







x = np.random.rand(500, 500, 3)

plt.imshow(x)

import cv2

img = cv2.imread("C:/PyTorch/lujiayi/CRN_flake/train/2020.02.27.11.42.36_L_1.bmp")

plt.imshow(img)

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img2)

img2.shape

mask = np.zeros(img2.shape)

mask[100:200,100:200,0] = 1

plt.imshow(mask)

plt.imshow(mask + img2)

img3 = img2
img3[mask == 1] = 255

plt.imshow(img3)



mask = np.zeros(img2.shape[0:2])
mask[100:200,100:200] = 1

mask.shape

img3 = img2.copy()
img3[mask == 1, 0] = 255
plt.imshow(img3)

img3 = img2.copy()
img3[mask == 1, 1] = 255
plt.imshow(img3)

img3 = img2.copy()
img3[mask == 1, 2] = 100
plt.imshow(img3, alpha = 1)

plt.imshow(img3)


plt.imshow(img2)
plt.imshow(mask, alpha = 0.2, cmap = 'gray')
plt.show()

plt.figure(figsize=(10,10))
plt.imshow(img2)
plt.imshow(mask, alpha = 0.5, cmap = 'jet')
plt.show()

