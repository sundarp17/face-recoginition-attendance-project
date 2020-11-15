import cv2
import numpy as np
import face_recognition
import os

#loading images and converting them into rgb
img_musk = face_recognition.load_image_file('imagebasic/musk_train.jpg')
img_musk = cv2.cvtColor(img_musk, cv2.COLOR_BGR2RGB)

img_musk_test = face_recognition.load_image_file('imagebasic/musk_test.jpg')
img_musk_test = cv2.cvtColor(img_musk_test, cv2.COLOR_BGR2RGB)

#finding the location and encodings
faceLoc = face_recognition.face_locations(img_musk)[0]
encodeElon = face_recognition.face_encodings(img_musk)[0]
cv2.rectangle(img_musk,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLoc_test = face_recognition.face_locations(img_musk_test)[0]
encodeElon_test = face_recognition.face_encodings(img_musk_test)[0]
cv2.rectangle(img_musk_test,(faceLoc_test[3],faceLoc_test[0]),(faceLoc_test[1],faceLoc_test[2]),(255,0,255),2)


#using linear svm to know whether they match or not

results = face_recognition.compare_faces([encodeElon],encodeElon_test)
#if there are similar images, we need to calculate the distance for best match...less distance meansmore mattching.
faceDist = face_recognition.face_distance([encodeElon],encodeElon_test)
print(results,faceDist)
cv2.putText(img_musk_test,f'{results} {round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('elon musk', img_musk)
cv2.imshow('elon musk test', img_musk_test)
cv2.waitKey(0)

