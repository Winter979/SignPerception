import cv2
import os
import numpy as np

from Settings import Settings as s

from Tools import *


def Find_Numbers(mask,image):
   cnts = Get_Contours(mask)

   letters = []

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      
      # Get crop and resize
      crop = mask[y:y+h,x:x+w]
      crop = cv2.resize(crop,(50,50))

      guess = Test_Number(crop)

      if guess != '?':
         letters.append([guess,x,y])
         cv2.rectangle(image, (x-2,y-2),(x+w+2,y+h+2), (0,0,255), 1)

   # Sort by X
   letters = sorted(letters, key=lambda x: x[1])

   number = "".join([ii[0] for ii in letters])

   return number

def Remove_Negatives(mask):
   cnts = Get_Contours(mask)
   
   new = np.zeros(mask.shape[:2],dtype="uint8")

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      area = w*h
      
      area2 = cv2.contourArea(c)

      rect = cv2.minAreaRect(c)
      (x1,x2),(w1,h1),angle = rect

      angle = abs(angle)

      if 200 < area  and area2 > 100:
         ratio = h/w
         if 1 < ratio < 4:
            if angle < 30 or angle > 60:
               cv2.drawContours(new,[c],-1,255,-1)

   better = cv2.bitwise_and(new,mask)

   return better
   
def Get_Mask(image):

   image = cv2.bilateralFilter(image, 50,75,75)

   gray,hsv = Cvt_All(image)

   # lower = np.array([0,0,100])
   # upper = np.array([255,255,255])
   # defs_dark = cv2.inRange(hsv, lower, upper)

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         cv2.THRESH_BINARY,15,-5)

   mask = th3.copy()


   # Didnt really work out to nicely
   # white = Create_Mask(hsv, "white")
   # mask = cv2.bitwise_and(mask,white)

   gold = Create_Mask(hsv,"gold")
   gold = cv2.medianBlur(gold, 21)
   mask = cv2.bitwise_and(mask,~gold)

   red = Create_Mask(hsv,"red")
   red = cv2.medianBlur(red, 21)
   mask = cv2.bitwise_and(mask,~red)

   Trim_Edges(mask,color=0)

   return mask




def Main(files):
   answers = Setup_Verifier("Building")

   for f in files:

      fname = os.path.basename(f).split(".")[0]

      image = cv2.imread(f)

      mask = Get_Mask(image)

      better = Remove_Negatives(mask)

      number = Find_Numbers(better,image)
      answer = answers[fname]

      if answer == answer:
         print("{}{} : {} == {}{}".format(Colrs.CYAN,fname, answer, answer,Colrs.RESET))
      else:
         print("{}{} : {} != {}{}".format(Colrs.RED,fname, answer, answer, Colrs.RESET))

      cv2.imshow("mask",mask)
      cv2.imshow("better",better)
      cv2.imshow("image",image)
      cv2.waitKey(0)

if __name__ == "__main__":
   # files = glob.glob("val_BuildingSignage/*.jpg")
   files = glob.glob("val_DirectionalSignage/*.jpg")
   main(files)