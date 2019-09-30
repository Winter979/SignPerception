import glob
import cv2

import numpy as np

from Tools import *

from Settings import Settings

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

def Extract_Numbers(mask):
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
   blur = image

   blur = cv2.bilateralFilter(image,5,75,75)

   gray,hsv = Cvt_All(blur)

   th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 15 ,-5)

   inrange = cv2.inRange(hsv,(0,0,100),(255,70,255))

   mask = cv2.bitwise_and(inrange,th)

   return mask 

def Find_Numbers_1(results,mask,image):

   results = Sort_Results(results,"Y")
   results = Group_Results(results,"Y")

   better = []

   for res in results:
      if len(res) >= 3:
         better.append(res)

   # better = results

   # Its generally the 3 largest ones. 
   # better = results[-3:]
   # better = Sort_Results(better, "X")
   return better

def Find_Numbers_2(results,mask,image):
   pass

def Group_Numbers(results, mask, image):
   better = []

   if Settings.task == 1:
      better = Find_Numbers_1(results,mask,image)
   elif Settings.task == 2:
      better = Find_Numbers_2(results,mask,image)

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


def Sort_By_X(results):
   pass

def Main(files):
   for f in files:
      image = cv2.imread(f)

      mask = Create_Mask(image) 

      results = Extract_Numbers(mask)

      filtered = Group_Numbers(results,mask, image)

      for group in filtered:
         label = ""
         for res in group:
            x,y,w,h = res["shape"]
            label += res["letter"]
            cv2.rectangle(image, (x,y),(x+w,y+h), (0,0,255), 1)
         print(label)   

      # if Settings.task == 1:
      #    label = ""
      #    for res in filtered:
      #       x,y,w,h = res["shape"]
      #       label += res["letter"]
      #       cv2.rectangle(image, (x,y),(x+w,y+h), (0,0,255), 1)
      #    print(label)

      # elif Settings.task == 2:
      #    for g in filtered:
      #       label = ""
      #       for res in g:
      #          x,y,w,h = cv2.boundingRect(res[0])
      #          label += res[1]
      #          cv2.rectangle(image, (x,y),(x+w,y+h), (0,0,255), 1)
      #       print(label)

      cv2.imshow('image', image)
      cv2.imshow('gold', gold)
      cv2.imshow("mask",mask)
      # cv2.imshow("th",th)
      # cv2.imshow("ultimate",ultimate)
      # cv2.imshow('new', new)
      cv2.waitKey(0)
      cv2.destroyAllWindows()


if __name__ == "__main__":
   files = glob.glob("{}/*".format(s.T1_Images))
   Main(files)

    