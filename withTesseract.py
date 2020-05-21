import cv2
from darkflow.net.build import TFNet
import numpy as np
import time
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import os
from difflib import SequenceMatcher

predicted = 0
# def accuracy(a,b):
#     consider = SequenceMatcher(None, a, b).ratio()
#     if consider>0.7:
#         predicted = predicted + 1



option = {
    'model': 'cfg/tiny-yolo-voc-3.cfg',
    'load': 875,
    'threshold': 0.1,
    'gpu': 1.0
}

tfnet = TFNet(option)

#capture = cv2.VideoCapture('test3.mp4')
#capture = cv2.VideoCapture('http://192.168.0.108:8000/stream.mjpg')
#capture = cv2.VideoCapture('http://192.168.43.1:8080/video')
capture = cv2.VideoCapture('http://192.168.43.196:8080/video')
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]  #all this will do is generate boxes of different colours

name = 1

#find similarity between two characters
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    # frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
    # cv2.imshow('Gaussian Filter',frame)


    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            print(label)
            frame = cv2.rectangle(frame, tl, br, color, 7)
            frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

            # this is the prgramme to store object when it is detected
            roi = frame[result['topleft']['y']:result['bottomright']['y'],
                  result['topleft']['x']:result['bottomright']['x']]
            cv2.imshow('roi', roi)  # displaying cropped image
            nameofimage = str(name) + '.jpg'  # name of cropped image
            path = os.path.join("detected_rashes", nameofimage)  # path of cropped image

            #applying filters on roi (not giving good accuracy)
            #roi = cv2.GaussianBlur(cv2.medianBlur(roi,5), (5, 5), cv2.BORDER_DEFAULT)
            # cv2.imshow('Median blur then Gaussian Filter',roi)
            # roi = cv2.medianBlur(roi, 3) #median filter


            #segmentation test




            #fetching the text of number plate
            # text = pytesseract.image_to_string(roi)
            # if(text!=""):
            #     if (text == "HR26DK8337"):
            #         print(text+" exists")
            #     else:
            #         print(text+" access denied")


            # text = pytesseract.image_to_string(roi)
            # print(text)

            #find similarity between two characters
            #print(similar(text,"RJ20CD5030"))

            # consider = SequenceMatcher(None, text, "RJ20CD5030").ratio()
            # if consider > 0.7:
            #     predicted = predicted + 1
            #
            #
            # name += 1  # incrementing name
            # cv2.imwrite(path, roi)  # writing cropped image


        cv2.imshow('frame', frame)
        #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        cv2.destroyAllWindows()
        break

print(predicted)