import cv2
import numpy as np
import urllib.request
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
 
url = 'http://192.168.43.246/cam-hi.jpg'


cred = credentials.Certificate(r"C:\Users\aymoo\Downloads/test.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://trolleytally-default-rtdb.europe-west1.firebasedatabase.app'})
## firebase database refreence
ref = db.reference("/")

objects = ["bottle","toothbrush","book"]


 
cap = cv2.VideoCapture(url)
whT=320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile='coco.names'
classNames=[]
with open(classesfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
#print(classNames)
 
modelConfig = 'yolov3.cfg'
modelWeights= 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObject(outputs,im):
    hT,wT,cT = im.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                
                classIds.append(int(classId))
                confs.append(float(confidence))
                
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)

    print(indices)
   
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]

        print(classNames[classIds[i]])
        

        for j in objects:
            if classNames[classIds[i]] == j:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
                cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                print(i)
                ref.child("cart1").child("detection").set(j)
                ref.child("cart1").child("enableDetectingStatus").set(False)
                break;
               
 
while True:
    print(ref.child("cart1").child("enableDetectingStatus").get())
    if(ref.child("cart1").child("enableDetectingStatus").get() == False):
        continue
    
    img_resp=urllib.request.urlopen(url)
    
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)

    
    im = cv2.imdecode(imgnp,-1)
    
    sucess, img= cap.read()

    print(sucess)
    print(img)
        
    blob=cv2.dnn.blobFromImage(im,1/255,(whT,whT),[0,0,0],1,crop=False)

    
    net.setInput(blob)
    layernames=net.getLayerNames()
    data = net.getUnconnectedOutLayers()
    outputNames = [layernames[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    
    findObject(outputs,im)
 
    cv2.imshow('IMage',im)
    cv2.waitKey(1)
