'''
File: Tools.py
File Created: Monday, 30th September 2019 7:49:43 pm
Author: Jonathon Winter
-----
Last Modified: Tuesday, 1st October 2019 12:17:25 am
Modified By: Jonathon Winter
-----
Purpose: 
'''

import cv2
import numpy as np
import glob

import json

from Settings import Settings

class Colrs:
   RED      = "\033[1;31m"  
   BLUE     = "\033[1;34m"
   CYAN     = "\033[1;36m"
   GREEN    = "\033[0;32m"
   RESET    = "\033[0;0m"
   BOLD     = "\033[;1m"
   REVERSE  = "\033[;7m"

MASKS = {
   # BGR Masks
   "p_red"  : [np.array([  0,  0,150]) , np.array([ 30, 30,255])],
   "dark"   : [np.array([  0,  0,  0]) , np.array([125,125,125])],
   "p_green": [np.array([  0,200,  0]) , np.array([125,255,125])],
   # HSL Masks
   "pink"   : [np.array([100,  0,150]) , np.array([150, 30,200])],
   "gold"   : [np.array([ 10,100,100]) , np.array([ 80,255,255])],
   "white"  : [np.array([  0,  0,155]) , np.array([255, 100,255])],
   "red1"   : [np.array([  0, 80, 100]) , np.array([ 30,255,255])],
   "red2"   : [np.array([160, 80, 100]) , np.array([180,255,255])],
   "yellow" : [np.array([ 21, 60, 80]) , np.array([ 40,255,255])],
   "green"  : [np.array([ 41, 60, 64]) , np.array([ 90,255,255])],
   "brown"  : [np.array([  5,100, 20]) , np.array([ 30,255,255])],
   "shadow" : [np.array([  0,110,  0]) , np.array([ 60,255, 80])],
   "sky"    : [np.array([100, 60,200]) , np.array([140,255,255])],
   "dark"   : [np.array([  0, 60,200]) , np.array([255,255,255])],
   "red"    :[[np.array([  0, 80, 100]) , np.array([ 30,255,255])],
             [np.array([160, 80, 100]) , np.array([180,255,255])]],
}

def Dilate(image, size):
   kernel = np.ones(size, np.uint8) 
   dilated = cv2.dilate(image, kernel, iterations=1) 

   return dilated


def Draw_Findings(image, results):
   
   ii = 0
   for res in results:
      x,y,w,h = res["shape"]
      cv2.imshow("letter-{}".format(ii),image[y:y+h,x:x+w])

      ii += 1

   x,y,w,h = results[0]["shape"]

   x1 = x
   x2 = x + w
   y1 = y
   y2 = y + h

   for res in results:
      x,y,w,h = res["shape"]

      x1 = x if x < x1 else x1
      x2 = x + w if x + w > x2 else x2
      y1 = y if y < y1 else y1
      y2 = y + h if y+h > y2 else y2


   if Settings.task == 1:
      buf = 10
   else:
      buf = 0

   print("DRAW")

   cv2.imshow("Number Group",image[y1-buf:y2+buf,x1-buf:x2+buf])
   cv2.rectangle(image, (x1-buf,y1-buf),(x2+buf,y2+buf),(0,0,255),1)

def HSV_Mask(image, color):

   if color == "red":
      lower, upper = MASKS[color][0]
      mask1 = cv2.inRange(image, lower, upper)

      lower, upper = MASKS[color][1]
      mask2 = cv2.inRange(image, lower, upper)

      mask = cv2.bitwise_or(mask1, mask2)
   else:
      lower, upper = MASKS[color]
      mask = cv2.inRange(image, lower, upper)


   # mask = cv2.medianBlur(mask, 7)

   Trim_Edges(mask, color=0)

   return mask


def Trim_Edges(mask, color = 255, width=1):
   pack = mask.shape

   h,w = pack[:2]

   for x in range(w):
      for ii in range(width):
         mask[ii,x] = color

   for x in range(w):
      for ii in range(width):
         mask[h-1-ii,x] = color

   for y in range(h):
      for ii in range(width):
         mask[y,ii] = color

   for y in range(h):
      for ii in range(width):
         mask[y,w-1-ii] = color


def Cvt_All(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   return gray, hsv

def Get_Contours(mask):
   res = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   return cnts


def Create_Empty(image,color=0):
   new = np.ones(image.shape[:2],dtype="uint8") * color
   
   return new

def Setup_Verifier():
   
   with open("./Answers.json") as f:
      data = json.load(f)   

   if Settings.task == 1:
      category = "Building"
   elif Settings.task == 2:
      category = "Directional"

   return data[category]