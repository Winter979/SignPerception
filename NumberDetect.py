'''
File: NumberDetect.py
File Created: Monday, 30th September 2019 10:09:23 pm
Author: Jonathon Winter
-----
Last Modified: Monday, 30th September 2019 11:29:34 pm
Modified By: Jonathon Winter
-----
Purpose: 
'''


import cv2
import numpy as np
from Settings import Settings

import glob

def What_Is_It(c,mask):
   c = c.copy()
   x,y,w,h = cv2.boundingRect(c)

   # print(guess)

   new = np.zeros((h,w),dtype="uint8")
   
   c -= (x,y)
   cv2.drawContours(new, [c],0,255,-1)
   new = cv2.resize(new,(50,50))

   crop = mask[y:y+h,x:x+w]
   crop = cv2.resize(crop,(50,50))

   new = cv2.bitwise_and(crop,new)

   guess = Test_Number(new)

   # if guess != '?':
   #    print(guess)
   #    cv2.imshow("new",new)
   #    cv2.waitKey(0)

   return guess


def Is_Arrow(r1,r2,a):
   if 0.8<r1<1.2:
      if 0.8<r2<1.2:
         if 35 < abs(a) < 55:
            return True

   return False

def Is_Number(ratio):

   if 1.15 < ratio < 4:
      return True

   return False

def Large_Enough(area):
   if Settings.task == 1:
      return 350 < area < 3000
   elif Settings.task == 2:
      return 100 < area < 1500

   return False

def Is_It_Something(c):
   x,y,w,h = cv2.boundingRect(c)
   (x1,y1),(w1,h1),angle = cv2.minAreaRect(c)   

   area = w*h
   area2 = w1*h1

   if not Large_Enough(area):
      return False

   angle = abs(angle)

   ratio = h/w
   ratio2 = w1/h1

   if Is_Arrow(ratio,ratio2,angle):
      return True

   if Is_Number(ratio):
      return True

   return False

def mse(imageA, imageB):
	score = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	score /= float(imageA.shape[0] * imageA.shape[1])

	return score

def Test_Number(number):

   files = glob.glob("templates/*.jpg")

   guesses = []

   for f in files:
      mask = cv2.imread(f)
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

      val = f[10]
      score = mse(number,mask)
      
      guesses.append([val,score])

   guesses = sorted(guesses, key=lambda x: x[1])


   if guesses[0][1] > 15000:
      return '?'
   else:
      return guesses[0][0]