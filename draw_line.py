# https://github.com/jiayi9/image/blob/master/cv2_line_example.png

import cv2
import numpy as np
import PIL
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import pylab

points = [
        "1175, 880",
        "1175, 880",
        "1175, 882",
        "1177, 890",
        "1180, 890",
        "1182, 890",
        "1192, 887",
        "1205, 880",
        "1210, 877",
        "1215, 875",
        "1242, 860",
        "1250, 850",
        "1252, 842",
        "1257, 835",
        "1257, 830",
        "1257, 822",
        "1257, 820",
        "1250, 800",
        "1245, 782",
        "1242, 775",
        "1235, 762",
        "1232, 755",
        "1227, 745",
        "1220, 735",
        "1217, 725",
        "1215, 722",
        "1210, 720",
        "1205, 717",
        "1197, 717",
        "1190, 717",
        "1175, 717",
        "1152, 725",
        "1130, 740",
        "1115, 750",
        "1107, 765",
        "1107, 790",
        "1107, 810",
        "1107, 822",
        "1120, 832",
        "1132, 842",
        "1132, 845",
        "1140, 852",
        "1150, 855",
        "1170, 847",
        "1210, 795",
        "1212, 757",
        "1212, 755",
        "1190, 755",
        "1172, 770",
        "1180, 815",
        "1232, 852",
        "1237, 840",
        "1235, 802",
        "1215, 792",
        "1172, 787",
        "1155, 812",
        "1155, 830",
        "1175, 830",
        "1205, 805",
        "1212, 780",
        "1197, 772",
        "1195, 772",
        "1170, 772",
        "1155, 807",
        "1152, 817",
        "1150, 817",
        "1150, 812",
        "1147, 810",
        "1142, 797",
        "1140, 787",
        "1137, 780",
        "1135, 772",
        "1132, 755",
        "1132, 752",
        "1132, 747",
        "1140, 745",
        "1157, 742",
        "1170, 740",
        "1177, 740",
        "1190, 740",
        "1195, 740",
        "1200, 740",
        "1202, 740",
        "1217, 750",
        "1227, 755",
        "1235, 760",
        "1240, 767",
        "1247, 775",
        "1255, 782",
        "1260, 795",
        "1265, 802",
        "1265, 800",
        "1265, 790",
        "1265, 782",
        "1265, 772",
        "1262, 767",
        "1262, 765",
        "1262, 762",
        "1257, 755",
        "1255, 755",
        "1247, 747",
        "1245, 742",
        "1240, 742",
        "1235, 740",
        "1222, 737",
        "1212, 732",
        "1200, 727",
        "1192, 727",
        "1177, 727",
        "1170, 727",
        "1170, 730",
        "1165, 730",
        "1160, 732",
        "1155, 735",
        "1140, 737",
        "1130, 740",
        "1122, 742",
        "1105, 745",
        "1097, 747",
        "1090, 750",
        "1087, 755",
        "1085, 757",
        "1082, 765",
        "1082, 770",
        "1082, 775",
        "1082, 780",
        "1082, 787",
        "1082, 795",
        "1085, 797",
        "1085, 805",
        "1085, 812",
        "1090, 827",
        "1092, 835",
        "1095, 842",
        "1102, 855",
        "1105, 860",
        "1110, 875",
        "1117, 887",
        "1117, 890",
        "1120, 892",
        "1122, 892",
        "1127, 892",
        "1135, 892",
        "1142, 892",
        "1145, 892"
      ]

# as polygon, fill = 1
stroke = 50
mask = np.zeros((2000, 2000), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)
xy = [point.split(", ") for point in points]
xy = [tuple([int(point[0]), int(point[1])]) for point in xy]
draw.polygon(xy=xy, fill=1, width=stroke)
plt.imshow(np.array(mask))
pylab.show()

# as line, fill = 0
stroke = 50
fill = 0
mask = np.zeros((2000, 2000), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)
xy = [point.split(", ") for point in points]
xy = [tuple([int(point[0]), int(point[1])]) for point in xy]
draw.line(xy=xy, fill=fill, width=stroke)
plt.imshow(np.array(mask))
pylab.show()

# as line, fill = 1, stroke = 50
stroke = 50
fill = 1
mask = np.zeros((2000, 2000), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)
xy = [point.split(", ") for point in points]
xy = [tuple([int(point[0]), int(point[1])]) for point in xy]
draw.line(xy=xy, fill=fill, width=stroke)
plt.imshow(np.array(mask))
pylab.show()


# as line, fill = 1, stroke = 20
stroke = 20
fill = 1
mask = np.zeros((2000, 2000), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)
xy = [point.split(", ") for point in points]
xy = [tuple([int(point[0]), int(point[1])]) for point in xy]
draw.line(xy=xy, fill=fill, width=stroke)
plt.imshow(np.array(mask))
pylab.show()

# as line, fill = 1, stroke = 30
stroke = 30
fill = 1
mask = np.zeros((2000, 2000), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)
xy = [point.split(", ") for point in points]
xy = [tuple([int(point[0]), int(point[1])]) for point in xy]
draw.line(xy=xy, fill=fill, width=stroke)
plt.imshow(np.array(mask))
pylab.show()


# as line, fill = 1, stroke = 10
stroke = 10
fill = 1
mask = np.zeros((2000, 2000), dtype=np.uint8)
mask = PIL.Image.fromarray(mask)
draw = PIL.ImageDraw.Draw(mask)
xy = [point.split(", ") for point in points]
xy = [tuple([int(point[0]), int(point[1])]) for point in xy]
draw.line(xy=xy, fill=fill, width=stroke)
plt.imshow(np.array(mask))
pylab.show()
