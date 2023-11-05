# -*- codeing = utf-8 -*-
# @Time : 2023/11/4 15:15
# @Author : xupan
# @File : facePro.py
# @Software:PyCharm


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'image'
images = []
names = []
myList = os.listdir(path)
print(myList)

def getAllName(list):
    for cl in list:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        names.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        return  encodeList

def markAttendance(name):
    with open('attandance.csv', 'r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
        dtstring = now.strftime('%H:%M:%S')
        f.writelines(f'\n{name},{dtstring}')


def toolmain():
    getAllName(myList)
    print(names)
    encodelist = findEncodings(images)
    print('encoding complete!')

    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # print(facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodelist, encodeFace)
            faceDis = face_recognition.face_distance(encodelist, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = names[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)

        cv2.imshow('cam', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    toolmain()