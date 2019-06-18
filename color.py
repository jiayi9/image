import numpy as np
import matplotlib.pyplot as plt
from skimage import io

palette = np.array([[255,   0,   0], # index 0: red
                        [  0, 255,   0], # index 1: green
                         [  0,   0, 255], # index 2: blue
                         [255, 255, 255], # index 3: white
                         [  0,   0,   0], # index 4: black
                         [255, 255,   0], # index 5: yellow
                         ], dtype=np.uint8)


io.imshow(np.array([0]))

io.imshow(np.array([[0], [255]]))

io.imshow(np.array([0, 255]))

io.imshow(np.array([[0, 0, 0], [255, 0 , 255]]))

io.imshow(np.array([[0, 0, 0], [255, 0 , 255]], dtype = np.uint8))


io.imshow(np.array([
        [ [0, 0 ,255], [0, 255,0], [255, 0, 0] ],
        
        [ [255,0,255] , [255, 255,0], [0, 255, 255] ],

        [ [255,255,255] , [0, 0, 0], [25, 21, 155] ]
        
        ]))

io.imshow(np.array([
        [ [0, 0 ,255], [0, 255,0], [255, 0, 0] ],
        
        [ [255,0,255] , [255, 255,0], [0, 255, 255] ],

        [ [255,255,255] , [0, 0, 0], [25, 21, 155] ]
        
        ], dtype = np.uint8)
)
