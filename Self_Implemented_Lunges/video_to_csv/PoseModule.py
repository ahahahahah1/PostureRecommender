# This code expects the input videos to be present inside this directory structure:
#           video_to_csv
#                 |
#                 |
#           input_videos
#           /           \
#          /             \
#        test           train
#       /  |  \        /  |  \
#      /   |   \      /   |   \
#     b1   b2   g1   b1   b2  g1
#  <------       videos    ------->

# And arranges their corresponding csv files into this directory structure
#           video_to_csv
#                 |
#                 |
#         csv(combined_reps)
#           /           \
#          /             \
#        test           train
#       /  |  \        /  |  \
#      /   |   \      /   |   \
#     b1   b2   g1   b1   b2  g1
#  <------      csv      ------->



import cv2

import mediapipe as mp

import time

import math

import csv
from os import walk
import pandas as pd

from pathlib import Path

#[2:07 PM, 12/31/2021] Gautam Chauhan Nisheeth Students Mtech: self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)
#[2:07 PM, 12/31/2021] Gautam Chauhan Nisheeth Students Mtech: Change this and it will work

class poseDetector():



    def __init__(self, mode=False, upBody=False, smooth=True,

                 detectionCon=0.5, trackCon=0.5):



        self.mode = mode

        self.upBody = upBody

        self.smooth = smooth

        self.detectionCon = detectionCon

        self.trackCon = trackCon



        self.mpDraw = mp.solutions.drawing_utils

        self.mpPose = mp.solutions.pose
# test diff values of this param for robustness, maybe always remove first and last peaks
        self.pose =self.mpPose.Pose(model_complexity=2,smooth_landmarks = True,min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)# self.mpPose.Pose(~self.mode, self.upBody, self.smooth,     self.detectionCon, self.trackCon~)



    def findPose(self, img, draw=True):   # THIS draw is responsible for drawing stick figure

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(imgRGB)
        # print(self.results)
        if self.results.pose_landmarks:

            if draw:
                # print(self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,

                                           self.mpPose.POSE_CONNECTIONS)

        # print("pose worls landmarks")
        # self.mpDraw.plot_landmarks(
        # self.results.pose_world_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img



    def findPosition(self, img, draw=True):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = .5
        fontColor              = (255,255,255)
        thickness              = 2
        lineType               = 2
        self.lmList = []

        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):

                h, w, c = img.shape

                
                if lm.visibility <.3:
                    lm.x=lm.y=lm.z=-1
                print(id, " = ",lm.visibility)
                # if id == 25:
                #     print(id, lm)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)

                self.lmList.append([id, lm.x, lm.y, lm.z])

                if draw:
                    cv2.putText(img,str(id),    (cx,cy),     font,     fontScale,    fontColor,    thickness,    lineType)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        # print(self.results.pose_landmarks)

        # if self.results.pose_world_landmarks:
        #     print("pose owrls")
        #     for id, lm in enumerate(self.results.pose_world_landmarks.landmark):

        #         h, w, c = img.shape

        #         # print(id, lm)

        #         cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)

        #         self.lmList.append([id, lm.x, lm.y, lm.z])

        #         if draw:

        #             cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList





import sys
import os
def main(path,filename,draw = False):
    
    file_path = "./input_videos/" + path + filename + '.mp4'
    if not os.path.isfile(file_path) :
        file_path = './input_videos/' + path + filename + '.MOV'
        if not os.path.isfile(file_path):
            file_path = './input_videos/'+ path + filename + '.mov'
    # print("input file :", file_path)

    cap = cv2.VideoCapture(file_path)
    p = time.time()
    detector = poseDetector()
    dic = {}
    

    a_file = Path('./csv(combined_reps)/' + path + filename + '.csv')
    a_file.parent.mkdir(exist_ok=True, parents=True)
    a_file = open('./csv(combined_reps)/' + path + filename + '.csv', "w+")

    i = 0

    writer = csv.writer(a_file, lineterminator='\n')

    while True:
        success, img = cap.read()

        if(not success):
            break

        i = i+1
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        normalized_vectors = [("time: ", time.time() - p, "", "format: id, x, y, z")]

        for pt in lmList:

            # if(pt[0]>=13 and pt[0]<=22):

            #     normalized_vectors.append((0, 0, 0))

            #     continue

            normalized_vectors.append((pt[0], pt[1], pt[2], pt[3]))

        dic[i] = normalized_vectors

        # r = 1000.0 / img.shape[0]
        # dim = ( int(img.shape[1] * r),1000)
        # # perform the actual resizing of the image
        # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        scale_percent = 60 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
                
        # ##resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow("Image", resized)
        # cv2.waitKey(0)

    for key, value in dic.items():

        arr = [key, ]

        for i in value:
            arr.append(i[0])
            arr.append(i[1])
            arr.append(i[2])
            arr.append(i[3])
            
            # print(len(arr),i[3])
        # print(arr,len(arr),"size")
        
        if len(arr) > 5:
            writer.writerow(arr)
        # print(arr)
    a_file.close()
    return



if __name__ == "__main__":
    print("Creating csv data from videos (inside the file PoseModule)") 
    f = []
    paths = []
    
    ''' 
        f contains the list of all filenames for all the videos (test and train)
        paths contains the relative filepath to that file consider the root as './input_videos/'
    '''
    
    for (root, dirs, filenames) in walk('./input_videos/'):
        for dir in dirs:
            for (label_dir, _, filenames) in walk(os.path.join(root, dir)):
                for filename in filenames:
                    # print("current filename = ", filename, "and label = ", label_dir[15:])
                    f.append(filename)
                    paths.append(label_dir[15:] + "/") # We are only storing the file path after './input_videos'
        break
    # print(f)
    # print(paths)
    for (path,fname) in zip(paths,f):
        main(path,fname[:-4],True)