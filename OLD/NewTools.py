import cv2
import numpy as np
import imutils

from Tools import *

from Masks import *

def Apply_Mask_Image(image, mask):
   new = image.copy()

   new[np.where(mask != 0)] = [255,255,255]

   return new

def Better_Mask(image):

   blur = image
   blur = cv2.bilateralFilter(image, 15,90,90)
   gray, hsv = Cvt_All(image)

   # Black Mask
   _,f1 = cv2.threshold(blur, 120,255, cv2.THRESH_BINARY)
   black = f1.copy()
   black = cv2.cvtColor(black, cv2.COLOR_BGR2GRAY)
   black[np.array(black != 0)] = 255

   # black = Remove_Noise(~black)

   MASKS = [
            Red_Mask,
            Green_Mask,
            Yellow_Mask,
            White_Mask,
            Brown_Mask,
            Black_Mask
            ]

   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   # Red Mask

   new_mask = black

   for func in MASKS:
      mask = func(hsv)
      new_mask += mask

   # red_mask = Red_Mask(hsv)
   # green_mask = Green_Mask(hsv)
   # yellow_mask = Yellow_Mask(hsv)
   # black_mask = Black_Mask(hsv)

   # new_mask = black + red_mask + green_mask + yellow_mask + black_mask
   new_mask[np.array(new_mask != 0)] = 255

   # Remove minor noise
   new_mask = cv2.medianBlur(new_mask, 7)

   return new_mask

def Remove_Noise(mask, image):

   canny = cv2.Canny(image, 200,200)

   

   # cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   # cnts = imutils.grab_contours(cnts)
   # cv2.drawContours(image, cnts, -1, (0,255,0), 5)

   # Gets rid of small-mid noise
   # mask = cv2.medianBlur(mask, 7)
   
   # mask = ~mask

   cv2.imshow("test", mask)
   cv2.imshow("canny", canny)

   return mask

def Filterd_Mask(image):

   ret,f1 = cv2.threshold(image, 120,255, cv2.THRESH_BINARY)
   mask = Find_Black(f1)

   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   f2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
   f3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

   f2 = cv2.medianBlur(f2,5)
   new = ~f2 & mask

   return new

   # canny = cv2.Canny(new, 220,230)

   # cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   # cnts = imutils.grab_contours(cnts)

   # cv2.drawContours(image, cnts, -1, (0,255,0), 2)

def Find_Black(image):

   image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   lower_white = np.array([0,0,0])
   upper_white = np.array([360,120,130])
   mask = cv2.inRange(image, lower_white, upper_white)

   return ~mask

