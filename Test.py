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
            results.append([c,letter])
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

   th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 9 ,-1)

   inrange = cv2.inRange(hsv,(0,0,100),(255,70,255))

   mask = cv2.bitwise_and(inrange,th)

   return mask 

def Group_Numbers(results, mask, image):
   better = []

   if Settings.task == 1:
      by_h = []
      for r in results:
         s = cv2.boundingRect(r[0])
         by_h.append([s,r])

      by_h = sorted(by_h, key=lambda x: x[0][3], reverse=True)

      largest = by_h[:3]
      better = [ii[1] for ii in largest]


   elif Settings.task == 2:
      by_y = []
      for r in results:
         s = cv2.boundingRect(r[0])
         by_y.append([s,r])

      by_y = sorted(by_y, key=lambda x: x[0][1])

      groups = []
      g_id = 0
      y_rng = 10

      groups.append([by_y[0]])
      for ii in range(1,len(by_y)):
         curr = by_y[ii][0][1]
         prev = by_y[ii-1][0][1]

         if abs(curr - prev) > y_rng:
            g_id += 1
            groups.append([]) 

         groups[g_id].append(by_y[ii])

      good_groups = []

      for g in groups:

         g = sorted(g,key=lambda x: x[0][0])

         g = [ii[1] for ii in g]
         
         if len(g) >= 4:
            good_groups.append(g)
         elif len(g) == 3:
            # missing = Find_Missing(g,image,mask)
            good_groups.append(g)
      better = good_groups
      # better = [ii[1] for ii in good_groups[0]]

   return better


def Main(files):
   mser = cv2.MSER_create()

   for f in files:
      image = cv2.imread(f);

      mask = Create_Mask(image) 

      results = Extract_Numbers(mask)

      filtered = Group_Numbers(results,mask, image)

      if Settings.task == 1:
         label = ""
         for res in filtered:
            x,y,w,h = cv2.boundingRect(res[0])
            cv2.rectangle(image, (x,y),(x+w,y+h), (0,0,255), 1)

         print(label)
      elif Settings.task == 2:
         for g in filtered:
            label = ""
            for res in g:
               x,y,w,h = cv2.boundingRect(res[0])
               label += res[1]
               cv2.rectangle(image, (x,y),(x+w,y+h), (0,0,255), 1)
            print(label)

      cv2.imshow('image', image)
      # cv2.imshow('blur', blur)
      cv2.imshow("mask",mask)
      # cv2.imshow("th",th)
      # cv2.imshow("ultimate",ultimate)
      # cv2.imshow('new', new)
      cv2.waitKey(0)
      cv2.destroyAllWindows()


if __name__ == "__main__":
   files = glob.glob("{}/*".format(s.T1_Images))
   Main(files)

    