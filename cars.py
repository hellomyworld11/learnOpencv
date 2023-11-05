# -*- codeing = utf-8 -*-
# @Time : 2023/10/10 21:10
# @Author : xupan
# @File : cars.py.py
# @Software:PyCharm

import cv2
import numpy as np

minw = 90
minh = 90

def center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return (cx, cy)

cap = cv2.VideoCapture('video.mp4')

line_high = 600
offset = 7
carnum = 0
cars = []
#去背景
bgsubmog = cv2.createBackgroundSubtractorKNN()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

while True:
    ret, frame = cap.read()

    if(ret == True):

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(frame, (7, 7), 5)

        mask = bgsubmog.apply(blur)

        erode = cv2.erode(mask, kernel)

        dilate = cv2.dilate(erode, kernel, iterations = 2)

        close = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)

        cnts, h = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame, (10, line_high), (1200, line_high), (255, 255, 0))

        for(i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            isvaild = (w >= minw) and (h >= minh)
            if(not isvaild):
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cpoint = center(x, y, w, h)
            cars.append(cpoint)
            cv2.circle(frame, (cpoint), 5, (0, 0, 255), -1)
            for (x, y) in cars:
                if((y > line_high - offset) and ( y < line_high + offset)):
                    carnum += 1
                    cars.remove((x, y))
                    print(carnum)

        cv2.putText(frame, "cars num:" + str(carnum), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),5)
        cv2.imshow('video', frame)
    #    cv2.imshow('video', close)

    key = cv2.waitKey(400)
    if(key == 27):
        break

cap.release()
cv2.destroyAllWindows()