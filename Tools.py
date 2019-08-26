import cv2
import glob
import numpy as np
import imutils

MASKS = {
   # BGR Masks
   "p_red"  : [np.array([  0,  0,150]) , np.array([ 30, 30,255])],
   "dark"   : [np.array([  0,  0,  0]) , np.array([125,125,125])],
   # HSL Masks
   "pink"   : [np.array([100,  0,150]) , np.array([150, 30,200])],
   "white"  : [np.array([  0,  0,200]) , np.array([255, 50,255])],
   "red1"   : [np.array([  0, 50, 80]) , np.array([ 30,255,255])],
   "red2"   : [np.array([160, 50, 80]) , np.array([180,255,255])],
   "yellow" : [np.array([ 21, 60, 80]) , np.array([ 40,255,255])],
   "brown"  : [np.array([  5,100, 20]) , np.array([ 30,255,200])],
   "green"  : [np.array([ 41, 60, 64]) , np.array([ 90,255,255])],
   "shadow" : [np.array([  0,110,  0]) , np.array([ 60,255, 80])]
}

def Gold_Mask(image):
   gray,hsv = Cvt_All(image)

   lower = np.array([10,100,100])
   upper = np.array([80,255,255])
   inrange = cv2.inRange(hsv, lower, upper)

   inrange = cv2.medianBlur(inrange, 5)

   cnts = cv2.findContours(~inrange, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
   cnts = imutils.grab_contours(cnts)

   for c in cnts:
      area = cv2.contourArea(c)

      if area < 1000:
         cv2.drawContours(inrange, [c], 0, 255, -1)
         
   inrange[np.where(inrange != 0)] = 255

   return inrange


def Create_Mask(image, color):

   lower, upper = MASKS[color]

   mask = cv2.inRange(image, lower, upper)
   mask = cv2.medianBlur(mask, 7)

   return mask

def Trim_Edges(mask, color = 255, width=3):
   h,w = mask.shape

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

def Apply_Mask_Image(image, mask):
   new = image.copy()

   new[np.where(mask != 0)] = [255,255,255]

   return new

def Cvt_All(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

   return gray, hsv