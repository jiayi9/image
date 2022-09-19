
import cv2
from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='ch') # need to run only once to download and load model into memory

video_capture = cv2.VideoCapture(0)
#video_capture.set(CV_CAP_PROP_FPS, 30)
while True:
    ret, frame = video_capture.read()

#     img_path = "C:/temp/temp.png"
#     cv2.imwrite(img_path, frame)
#     result = ocr.ocr(img_path, cls=True)

    result = ocr.ocr(frame, cls=True)

    from PIL import Image

#    image = Image.open(img_path).convert('RGB')
#    image = cv2.imread(img_path)



    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(frame, boxes, txts, scores, font_path='./fonts/simfang.ttf')


    cv2.imshow('Video', im_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
