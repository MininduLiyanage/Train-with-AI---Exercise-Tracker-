import numpy as np
import time
import cv2

import poseModule as pm

cap = cv2.VideoCapture('E:/EEE/comvis/ComputerVision/posedetection/gym3.mp4')

detector = pm.poseDetector()

count = 0
dir = 0

while True:
        success, img = cap.read()
        img = cv2.resize(img, (420,780))

        img =detector.findpose(img,False)

        lmlist= detector.getPosition(img, False)
        #print(lmlist)
        if len(lmlist)!=0:
                angle = detector.findAngle(img,11,13,15)#left
                #angle =detector.findAngle(img,12,14,16)#right
                per = np.interp(angle,(50,170),((100,0)))
                bar = np.interp(angle,(50,170),((480,780)))
                #print(angle,per)

                # track movements
                color = (0,255,255)
                if per ==100:
                        color = (0,255,0)
                        if dir == 0:
                               count+= 0.5
                               dir =1
                               
                if per == 0:
                        if dir == 1:
                               count+= 0.5
                               dir = 0
                               
                #print(per,dir,count)
                
                #display count
                cv2.rectangle(img, (0,0),(90,90),(0,255,0),cv2.FILLED)  
                cv2.putText(img,str(int(count)), (30,60),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,50),2)

                #dispaly bar
                cv2.rectangle(img, (30,780),(100,480),(0,255,0),3)  
                cv2.rectangle(img, (30,int(bar)),(100,780),color,cv2.FILLED)  
                cv2.putText(img,f"{int(per)}%", (35,440),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,color,2)

                #print(count)  

        cv2.imshow("Image",img)
        cv2.waitKey(10)