import cv2
import numpy as np
import requests
import urllib.request as rq

get_headers={'Api-Access-Key':'4AB8N3RS4JSD64FF7A030CO65H23LNAS13JD5478'}
post_headers={'Content-type':'multipart/form-data','Api-Access-Key':'4AB8N3RS4JSD64FF7A030CO65H23LNAS13JD5478'}
query={'emp_id':'DummySP321'}
response=requests.get('https://sitemployee.sourcesoftsolutions.com/services/getFaceImages',headers=get_headers,params=query)
emp_data=response.json()
print(emp_data)
print(emp_data['data']['emp_id'])
img_data=emp_data['data']['images']
print(len(img_data))

images=[]
for i in img_data:
    resp = rq.urlopen(i['url'])
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    curImg = cv2.imdecode(img,cv2.IMREAD_COLOR)
    images.append(curImg)

# --- import the Haar cascades for face and eye ditection
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
spec_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye_tree_eyeglasses.xml')

# FACE RECOGNISER OBJECT
LBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 20)
EIGEN = cv2.face.createEigenFaceRecognizer(10, 5000)
FISHER = cv2.face.createFisherFaceRecognizer(5, 500)

# Load the training data from the trainer to recognise the faces
LBPH.load(images)
#EIGEN.load("Recogniser/trainingDataEigan.xml")
#FISHER.load("Recogniser/trainingDataFisher.xml")

