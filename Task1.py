import cv2
import glob
import os
import sys
import numpy as np

from Settings import Settings as s

from Tools import *
def Get_White_Mask(image):
   _,mask = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
   
   # Convert mask to white & black
   binary = np.zeros(image.shape[:2],dtype="uint8")
   binary[np.where((mask == [0,0,0]).all(axis=2))] = 255

   return binary

def Better_Mask(image):
   gray,hsv = Cvt_All(image)

   mask = Get_White_Mask(image)

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         cv2.THRESH_BINARY,15,0)

   better = cv2.bitwise_and(mask,th3)



   return better
   
def Dark(image):

   image = cv2.bilateralFilter(image, 50,75,75)

   gray,hsv = Cvt_All(image)

   lower = np.array([0,0,100])
   upper = np.array([255,255,255])
   defs_dark = cv2.inRange(hsv, lower, upper)

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         cv2.THRESH_BINARY,5,0)

   return th3

def MSER_IT(image):
   mser = cv2.MSER_create()

   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   regions, _ = mser.detectRegions(gray)
   hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

   for c in hulls:
      x,y,w,h = cv2.boundingRect(c)
      area = w*h
      if 100 < area < 2000:
         ratio = h/w
         if 1.3 < ratio < 4:
            cv2.drawContours(image, [c], 0, (0,255,0),0)


   return image

def Find_Poss(mask):

   Trim_Edges(mask, color=0)

   res = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   new = np.zeros(mask.shape[:2],dtype="uint8")

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      area = w*h

      if 300 < area < 2000:
         ratio = h/w

         if 1.25<ratio < 3:
            cv2.drawContours(new, [c],0,255,-1)
         else:
            cv2.drawContours(new, [c],0,120,-1)


   return new


def main(files):
   for f in files:
      image = cv2.imread(f)

      dark = Dark(image)

      temp = dark.copy()
      temp = Find_Poss(temp)

      # white = Get_White_Mask(image)

      # new = cv2.bitwise_and(dark,white)

      # better = Better_Mask(image)
      # MSER_IT(image)

      # cv2.imshow("new",new)
      cv2.imshow("temp",temp)
      cv2.imshow("dark",dark)
      cv2.imshow("image",image)
      cv2.waitKey(0)

if __name__ == "__main__":
   files = glob.glob("BuildingSignage/*.jpg")
   main(files)