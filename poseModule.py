import cv2
import time
import mediapipe as mp
import math

class poseDetector():

    def __init__(self, mode = False, upperBody = False, smooth = True, detectionConf = 0.5, trackConf = 0.5):
    
        self.mode = mode
        self.upperBody = upperBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf


        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    # draw landmarks on the body
    def findpose(self,img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(imgRGB)

        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    # coordinates of each landmark of the skeleton
    def getPosition(self,img, draw=True):
        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                #print(id,lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)

        return self.lmlist

    # given 3 points of the skeleton, find angle between them
    def findAngle(self,img,p1,p2,p3, draw=True):
        
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]

        angle = 180 - math.degrees(math.atan2(y2-y3,x2-x3)-math.atan2(y1-y2,x1-x2))
        if angle<0:
            angle += 360
        #print(angle)

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,0),3)
            cv2.line(img,(x3,y3),(x2,y2),(0,0,0),3)
            cv2.circle(img, (x1,y1), 8, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x1,y1), 12, (255,0,0), 2)
            cv2.circle(img, (x2,y2), 8, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 12, (255,0,0), 2)
            cv2.circle(img, (x3,y3), 8, (255,0,0), cv2.FILLED)
            cv2.circle(img, (x3,y3), 12, (255,0,0), 2)
            cv2.putText(img,str(int(angle)), (x2-60,y2-40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)

        return angle
        

def main():

    cap = cv2.VideoCapture('E:/EEE/comvis/ComputerVision/posedetection/gym3.mp4')
    pTime = 0

    detector = poseDetector()

    while True:
        success, img = cap.read()

        img = detector.findpose(img)
        lmlist = detector.getPosition(img)

        if len(lmlist) != 0:
            print(lmlist[12])
            cv2.circle(img, (lmlist[12][1],lmlist[12][2]), 15, (0,255,0), cv2.FILLED)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'frame rate :{str(int(fps))}',(30,20), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,0),1)

        cv2.imshow("Image", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
