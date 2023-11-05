# -*- codeing = utf-8 -*-
# @Time : 2023/11/4 10:48
# @Author : xupan
# @File : basic.py
# @Software:PyCharm


import cv2
import numpy as np
import face_recognition

imgMask = face_recognition.load_image_file('image/ailong mask.jpg')
imgMaskTest = face_recognition.load_image_file('image/ailong mask test.jpg')
imgJackMaTest = face_recognition.load_image_file('image/jackma.jpg')

imgMask = cv2.cvtColor(imgMask, cv2.COLOR_BGR2RGB)
imgMaskTest = cv2.cvtColor(imgMaskTest, cv2.COLOR_BGR2RGB)
imgJackMaTest = cv2.cvtColor(imgJackMaTest, cv2.COLOR_BGR2RGB)

faceloc =  face_recognition.face_locations(imgMask)[0]
encodeMask = face_recognition.face_encodings(imgMask)[0]


facelocTest = face_recognition.face_locations(imgMaskTest)[0]
encodeMaskTest = face_recognition.face_encodings(imgMaskTest)[0]

facelocJackMa = face_recognition.face_locations(imgJackMaTest)[0]
encodeJackMa = face_recognition.face_encodings(imgJackMaTest)[0]

# faceloc保存的人脸位置 top right bottom left
cv2.rectangle(imgMask, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)
cv2.rectangle(imgMaskTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (255, 0, 255), 2)
cv2.rectangle(imgJackMaTest, (facelocJackMa[3], facelocJackMa[0]), (facelocJackMa[1], facelocJackMa[2]), (255, 0, 255), 2)

rets = face_recognition.compare_faces([encodeMask], encodeMaskTest)
facedis = face_recognition.face_distance([encodeMask], encodeMaskTest)
print('compare1:{}'.format(rets))
cv2.putText(imgMaskTest, f'{rets} {round(facedis[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

rets1 = face_recognition.compare_faces([encodeMask], encodeJackMa)
facedis1 = face_recognition.face_distance([encodeMask], encodeJackMa)
print('compare2:{}'.format(rets1))
cv2.putText(imgJackMaTest, f'{rets1} {round(facedis1[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)



cv2.imshow('mask template', imgMask)
cv2.imshow('mask test', imgMaskTest)
cv2.imshow('jackMa test', imgJackMaTest)
cv2.waitKey(0)