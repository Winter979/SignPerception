'''
File: Task.py
File Created: Monday, 30th September 2019 7:49:43 pm
Author: Jonathon Winter
-----
Last Modified: Tuesday, 1st October 2019 12:27:59 am
Modified By: Jonathon Winter
-----
Purpose: 
'''

import glob
import cv2

import os

import numpy as np

from Tools import *
from NumberDetect import *
from Settings import Settings


def Extract_Numbers(mask):
   mask = mask.copy()
   cnts = Get_Contours(mask)
   
   # new = np.zeros(mask.shape[:2],dtype="uint8")

   results = []

   for c in cnts:
      if Is_It_Something(c):
         letter = What_Is_It(c,mask)
         if letter != '?':
            result = {}

            result["contour"] = c
            result["letter"] = letter
            result["shape"] = cv2.boundingRect(c)
            results.append(result)
            # cv2.drawContours(new, [c],0,255,-1)
         # else:
         #    cv2.drawContours(new, [c],0,125,-1)

   # new = cv2.bitwise_and(new,mask)

   return results

def Clean(mask):
   clean = mask.copy()

   h,w = mask.shape[:2]

   rng = 1

   for x in range(rng,w-rng):
      for y in range(rng,h-rng):
         v1 = 0
         v2 = 0
         for ii in range(-rng,rng+1):
            v1 += int(mask[y+ii,x])
            v2 += int(mask[y,x+ii])

         if v1 <= (rng*255) or v2 <= (rng*255):
            # print(v)
            clean[y,x] = 0

   return clean

def Create_Mask(image):

   mask = Create_Empty(image,color=255)

   blur = image
   blur = cv2.bilateralFilter(image,5,75,75)

   gray,hsv = Cvt_All(blur)

   th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25 ,-10)

   # dilate = th
   dilate = Dilate(th,(5,1))
   # th = cv2.threshold(th,100,255,cv2.THRESH_BINARY)

   better = Create_Empty(image)
   cnts = Get_Contours(dilate)


   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)

      area = w*h

      if Is_It_Something(c):
         cv2.drawContours(better,[c],0,255,-1)
      # else:
      #    cv2.drawContours(th,[c],0,0,-1)

   better = np.bitwise_and(th,better)

   inrange = cv2.inRange(hsv,(0,0,100),(255,70,255))
   gold = HSV_Mask(hsv, "gold")
   gold = cv2.medianBlur(gold, 25)

   # Only accept if it is included in these 3 masks
   mask = cv2.bitwise_and(mask,better)
   mask = cv2.bitwise_and(mask,inrange)
   mask = cv2.bitwise_and(mask,~gold)
   
   return mask 

def Find_Numbers_1(results,image):

   results = Sort_Results(results,"Y")
   results = Group_Results(results,"Y")

   better = []

   for res in results:
      if len(res) >= 3:
         res = sorted(res, key=lambda x:x["shape"][0])

         better.append(res)

   # better = results

   # Its generally the 3 largest ones. 
   # better = results[-3:]
   # better = Sort_Results(better, "X")
   return better

def Find_Numbers_2(results,image):
   better = []

   results = Sort_Results(results,"Y")
   results = Group_Results(results,"Y")

   for res in results:
      # There arent enough. Ignore that Y group
      if len(res) <= 2:
         continue
      
      # Sort and group by X
      temp = Sort_Results(res,"X")
      temp = Group_Results(temp,"X",rng=50)

      for group in temp:
         if len(group) == 4:
            better.append(group)
         elif len(group) == 3:
            c, guess = Guess_Number(image, group)

            # cv2.drawContours(image,[c],0,(0,255,255),-1)

            new = {}

            new["contour"] = c
            new["letter"] = guess
            new["shape"] = cv2.boundingRect(c)

            group.insert(0,new)

            better.append(group)
      
      # for ii in range(len(res)):
      #    if res[ii]["shape"] in ["U","D","L","R"]:
      #       res = res[:ii]
      #       break
      
   # better.append(results)
   return better

def Guess_Number(image, known):
   x1,y1,w1,h1 = known[0]["shape"]
   x2,y2,w2,h2 = known[1]["shape"]

   x = x1 - (x2-x1)
   y = y1 + (y1-y2)

   h = (h1+h2)//2 +2
   w = int(0.7 * h) +2
   # cv2.rectangle(image, (x,y),(x+w,y+h),(255,255,0),1)

   cropI = image[y:y+h,x:x+w]

   c,guess = Find_Number(cropI)

   c += (x,y)

   return c,guess


def Find_Number(image):
   # gray,hsv = Cvt_All(image)

   gray,hsv = Cvt_All(image)

   th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 5 ,-5)

   Trim_Edges(th,color=0,width=1)

   cnts = Get_Contours(th)
   cnts = sorted(cnts,key=lambda x: cv2.contourArea(x),reverse=True)

   c = cnts[0]

   guess = What_Is_It(c,th)

   return c,guess

   # print(guess)



   # temp = np.zeros(image.shape[:2],dtype="uint8")
   # cv2.drawContours(temp, [c],0,255,-1)

   # better = cv2.bitwise_and(th,temp)
   


   # x,y,w,h = cv2.boundingRect(c)

   


   # cv2.imshow("cropI",image)
   # cv2.imshow("mask",th)
   # cv2.imshow("temp",temp)  
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
   
   

def Group_Numbers(results, image):
   better = []

   func = [Find_Numbers_1, Find_Numbers_2]

   better = func[Settings.task-1](results,image)
   
   return better

def Sort_Results(results,by):
   idx = { "X":0, "Y":1, "W":2, "H":3 }

   return sorted(results, key=lambda x: x["shape"][idx[by]])

def Group_Results(results,by,rng=10):
   idx = { "X":0, "Y":1, "W":2, "H":3 }

   groups = []

   groups.append([results[0]])
   prev = results[0]["shape"][idx[by]]
   for res in results[1:]:
      curr = res["shape"][idx[by]]
      if abs(curr - prev) > rng:
         groups.append([])

      groups[-1].append(res)
      prev = curr

   return groups


def Show_Answer(image,filtered, filename):

   answers = Setup_Verifier()
   
   fname = os.path.basename(filename).split(".")[0]
   answer = answers[fname]

   labels = []

   for group in filtered:
      label = ""

      # if Settings.show:
      #    Draw_Findings(image, group)

      for res in group:
         label += res["letter"]

      labels.append(label)
      
   if type(answer) != list:
      labels = labels[0]

   if Settings.test:
      if labels == answer:
         print("{}{} : {} == {}{}".format(Colrs.CYAN,fname, answer, labels,Colrs.RESET))
      else:
         print("{}{} : {} != {}{}".format(Colrs.RED,fname, answer, labels, Colrs.RESET)) 
   else:
      print(labels)

   if Settings.show:
      cv2.imshow('image', image)
      
      cv2.waitKey(0)
      cv2.destroyAllWindows()


def Main(files):

   for f in files:
      # Read the initial image
      image = cv2.imread(f)

      # A binary mask of the image with points of interest
      mask = Create_Mask(image) 

      # A list of numbers and their corresponding contours
      results = Extract_Numbers(mask) 

      # Put the numbers into groups relative to (x,y)
      filtered = Group_Numbers(results, image)

      # The expected answer
      Show_Answer(image, filtered, f)

if __name__ == "__main__":
   files = glob.glob("{}/*".format(Settings.T1_Images))
   Main(files)

    