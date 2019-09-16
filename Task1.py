import cv2
import glob
import os
import sys
import numpy as np

from Settings import Settings as s

from Tools import *

def Find_Possible(mask):
   res = cv2.findContours(~mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   # cnts = imutils.grab_contours(cnts)  
   
   new = np.ones(mask.shape[:2],dtype="uint8") * 255

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
            cv2.drawContours(new,[c],-1,0,-1)

   return new
   
def Get_Mask(image):

   image = cv2.bilateralFilter(image, 50,75,75)

   gray,hsv = Cvt_All(image)

   lower = np.array([0,0,100])
   upper = np.array([255,255,255])
   defs_dark = cv2.inRange(hsv, lower, upper)

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         cv2.THRESH_BINARY,15,-5)

   return th3

def Defs_Not(mask):

   Trim_Edges(mask, color=0)

   res = cv2.findContours(~mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   new = np.zeros(mask.shape[:2],dtype="uint8")

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      area = w*h

      if 200 < area < 2500:
         ratio = h/w

         if 1.2<ratio < 3:
            cv2.drawContours(new, [c],0,255,-1)
            print(area)
         else:
            pass
            # cv2.drawContours(new, [c],0,125,-1)


   return new

def Gold_Mask2(hsv,image):
   

   gold = Create_Mask(hsv, "gold")

   Trim_Edges(gold,color=0)

   gold = cv2.medianBlur(gold, 55)

   res = cv2.findContours(gold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   cnts,_ = res if len(res) == 2 else res[1:3]
   
   for c in cnts:
      area = cv2.contourArea(c)

      if area < 1000:
         cv2.drawContours(gold, [c], 0, 255, -1)
         # cv2.drawContours(image, [c], 0, (0,0,255), 2)
         
   return ~gold

def main(files):
   for f in files:
      image = cv2.imread(f)
      _,hsv = Cvt_All(image)

      white = Create_Mask(hsv, "white")


      mask = Get_Mask(image)
      maybe = Defs_Not(mask)

      gold = Gold_Mask2(hsv,image)

      new = cv2.bitwise_and(maybe,white)
      new = cv2.bitwise_and(new,new,mask=gold)

      cv2.imshow("white",white)
      cv2.imshow("gold",gold)

      cv2.imshow("new",new)
      cv2.imshow("maybe",maybe)
      cv2.imshow("mask",mask)
      cv2.imshow("image",image)
      cv2.waitKey(0)

if __name__ == "__main__":
   # files = glob.glob("val_BuildingSignage/*.jpg")
   files = glob.glob("val_DirectionalSignage/*.jpg")
   main(files)