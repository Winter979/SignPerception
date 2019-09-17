import cv2
import glob
import os
import sys
import numpy as np

from Settings import Settings as s

from Tools import *

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def Test_Number(number):
   files = glob.glob("templates/*.jpg")

   guesses = []

   for f in files:
      mask = cv2.imread(f)
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

      val = f[10]
      score = mse(number,mask)
      
      guesses.append([val,score])


      # cv2.imshow("number",number)
      # cv2.imshow("mask",mask)
      # cv2.waitKey()

   guesses = sorted(guesses, key=lambda x: x[1])

   if guesses[0][1] > 10000:
      return '?'
   else:
      return guesses[0][0]

def Find_Numbers(mask,image):
   res = cv2.findContours(~mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   letters = ""

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      
      # Get crop and resize
      crop = image[y:y+h,x:x+w]
      crop = cv2.resize(crop,(50,50))

      gray,_ = Cvt_All(crop)

      _,test = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      guess = Test_Number(test)

      if guess != '?':
         letters += guess

      # cv2.imshow("test",test)
      # cv2.waitKey(0)

      cv2.rectangle(image, (x-2,y-2),(x+w+2,y+h+2), (0,0,255), 1)

   # cv2.drawContours(image, cnts, -1, (0,255,0),1)

   print(letters)

def Find_Possible(mask):
   res = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            if angle < 30 or angle > 60:
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

      if 200 < area < 3000:
         ratio = h/w

         if 1.2<ratio < 3:
            cv2.drawContours(new, [c],0,255,-1)
         else:
            pass
            # cv2.drawContours(new, [c],0,125,-1)


   return new

def Gold_Mask2(hsv):
   

   gold = Create_Mask(hsv, "gold")

   Trim_Edges(gold,color=0)

   gold = cv2.medianBlur(gold, 55)

   res = cv2.findContours(gold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]
   
   for c in cnts:
      area = cv2.contourArea(c)

      if area < 1000:
         cv2.drawContours(gold, [c], 0, 255, -1)
         # cv2.drawContours(image, [c], 0, (0,0,255), 2)
         
   return ~gold

def Red_Mask(hsv):
   
   red1 = Create_Mask(hsv, "red1")
   red2 = Create_Mask(hsv, "red2")

   red = red1 + red2

   # Trim_Edges(red,color=0)

   # red = cv2.medianBlur(red, 55)

   # res = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   # cnts,_ = res if len(res) == 2 else res[1:3]
   
   # for c in cnts:
   #    area = cv2.contourArea(c)

   #    if area < 1000:
   #       cv2.drawContours(red, [c], 0, 255, -1)
         # cv2.drawContours(image, [c], 0, (0,0,255), 2)
         
   return ~red

def main(files):
   for f in files:
      image = cv2.imread(f)

      blur = cv2.bilateralFilter(image, 15,50,50)

      _,hsv = Cvt_All(blur)

      white = Create_Mask(hsv, "white")

      mask = Get_Mask(blur)
      maybe = Defs_Not(mask)

      gold = Gold_Mask2(hsv)
      red = Red_Mask(hsv)

      new = cv2.bitwise_and(maybe,white)
      new = cv2.bitwise_and(new,new,mask=gold)
      new = cv2.bitwise_and(new,new,mask=red)

      idk = Find_Possible(new)

      Find_Numbers(idk,image)

      cv2.imshow("idk",idk)
      cv2.imshow("image",image)
      cv2.waitKey(0)

if __name__ == "__main__":
   # files = glob.glob("val_BuildingSignage/*.jpg")
   files = glob.glob("val_DirectionalSignage/*.jpg")
   main(files)