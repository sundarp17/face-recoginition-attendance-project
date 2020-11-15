import cv2
import numpy as np
import face_recognition
from time import time
import requests
from datetime import datetime
import urllib.request as rq
import json
import base64
import argparse

images = []
names =[]

#ap = argparse.ArgumentParser()
#ap.add_argument("-d","--detection-method",type=str,default="cnn",
                #help="face detection model to use:either hog or cnn")
#args=vars(ap.parse_args())

get_headers={'Api-Access-Key':'4AB8N3RS4JSD64FF7A030CO65H23LNAS13JD5478'}
post_headers={'Content-type':'multipart/form-data','Api-Access-Key':'4AB8N3RS4JSD64FF7A030CO65H23LNAS13JD5478'}
query={'emp_id':'DummySP321'}
response=requests.get('https://sitemployee.sourcesoftsolutions.com/services/getFaceImages',headers=get_headers,params=query)
emp_data=response.json()
print(emp_data)
print(emp_data['data']['emp_id'])
img_data=emp_data['data']['images']
print(len(img_data))

#haar_cascade_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
for i in img_data:
    resp = rq.urlopen(i['url'])
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    curImg = cv2.imdecode(img,cv2.IMREAD_COLOR)
    images.append(curImg)
#cv2.imshow('dinesh',images[0])
#cv2.waitKey(1000)
def encodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        face_location = (0, width, height, 0)
        encode = face_recognition.face_encodings(img,known_face_locations=[face_location],num_jitters=10)[0]
        encodeList.append(encode)
    return encodeList
encoding_values= encodings(images)
print('encoding completed')

api='https://sitemployee.sourcesoftsolutions.com/services/saveAttendanceStatus'
emp_id = emp_data['data']['emp_id']
cam = cv2.VideoCapture(0)
previous = time()
diff = 0
while True:
    current = time()
    diff += current - previous
    previous = current

    if diff>10:
        success, img = cam.read()
        t = datetime.now()
        print(t.strftime('%H:%M:%S'))
        imgs = cv2.resize(img,(0,0),None,0.25,0.25)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
        #img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #faces= haar_cascade_face.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
        #print('Faces found: ', len(faces))
        faceLoc_current = face_recognition.api.face_locations(imgs,number_of_times_to_upsample=2,model="cnn")
        #print(faceLoc_current)
        if not faceLoc_current:
            is_present_noface=0
            payload_noface = json.dumps({"emp_id": emp_id, "is_present": is_present_noface})
            response_post_noface = requests.post(url=api, data=payload_noface, headers=post_headers)
            print(response_post_noface.text)

        encode_current = face_recognition.face_encodings(imgs,faceLoc_current,num_jitters=10)
        cv2.imshow('now',img)
        for encodeFace, Faceloc in zip(encode_current, faceLoc_current):
            matches = face_recognition.compare_faces(encoding_values,encodeFace,tolerance=0.6)
            distance = face_recognition.face_distance(encoding_values,encodeFace)
            print(distance)
            print(matches)
            if all(i==False for i in matches):
                success, buffer = cv2.imencode('.jpg',img) #encoding as .jpg
                img_text = base64.b64encode(buffer).decode("utf8")
                is_present = 0
                payload = json.dumps({"emp_id": emp_id, "is_present":is_present, "user_image":img_text})
                response_post = requests.post(url=api,data=payload,headers=post_headers)
                #print(img_text)
                print(response_post.text)
            matchIndex = np.argmin(distance)
            if matches[matchIndex]:
                name=emp_data['data']['first_name']
                y1,x2,y2,x1 = Faceloc
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255),2)
                is_present_true=1
                payload_true = json.dumps({"emp_id": emp_id, "is_present":is_present_true})
                response_post_true = requests.post(url=api, data=payload_true, headers=post_headers)
                print(response_post_true.text)

        diff =0
        cv2.imshow('webcam',img)
    cv2.waitKey(2)








