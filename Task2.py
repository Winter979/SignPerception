import cv2
import numpy as np

import os

from Settings import Settings as s

from Tools import *


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

      if 100 < area < 1100:
         ratio = h/w
         ratio2 = w1/h1
         if 1.25 < ratio < 4:
            cv2.drawContours(new,[c],-1,255,-1)
         elif Is_Arrow(ratio,ratio2,angle):
            cv2.drawContours(new,[c],-1,255,-1)
         else:
            pass
            # cv2.drawContours(new,[c],-1,100,-1)
            
   better = cv2.bitwise_and(new,mask)

   return better

def Is_Arrow(r1,r2,a):
   if 0.8<r1<1.2:
      if 0.8<r2<1.2:
         if 35 < abs(a) < 55:
            return True

   return False

def Get_Mask(image):

   image = cv2.bilateralFilter(image, 50,75,75)

   gray,hsv = Cvt_All(image)

   lower = np.array([0,0,100])
   upper = np.array([255,255,255])
   defs_dark = cv2.inRange(hsv, lower, upper)

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
         cv2.THRESH_BINARY,9,-1)

   mask = th3.copy()


   # Didnt really work out to nicely
   # white = Create_Mask(hsv, "white")
   # mask = cv2.bitwise_and(mask,white)

   green = Create_Mask(hsv,"green")
   green = cv2.medianBlur(green, 21)
   mask = cv2.bitwise_and(mask,~green)

   sky = Create_Mask(hsv,"sky")
   # sky = cv2.medianBlur(sky, 21)
   mask = cv2.bitwise_and(mask,~sky)

   gold = Create_Mask(hsv,"gold")
   gold = cv2.medianBlur(gold, 21)
   mask = cv2.bitwise_and(mask,~gold)

   red = Create_Mask(hsv,"red")
   red = cv2.medianBlur(red, 21)
   mask = cv2.bitwise_and(mask,~red)

   Trim_Edges(mask,color=0)

   return mask

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

def Find_Numbers(mask):
   cnts = Get_Contours(mask)

   letters = []

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)

      crop = mask[y:y+h,x:x+w]
      crop = cv2.resize(crop,(50,50))

      guess = Test_Number(crop)

      if guess != '?':
         letters.append([guess,x,y,c])
         # cv2.rectangle(image, (x-2,y-2),(x+w+2,y+h+2), (0,0,255), 1)

   # Sort by y
   letters = sorted(letters, key=lambda x: x[2])

   return letters

def Group_Letters(letters):
   groups = []

   y_rng = 10

   g_id = 0

   # Start the first group
   groups.append([letters[0]])

   # Skip the first one
   for ii in range(1,len(letters)):
      c = letters[ii][2]
      p = letters[ii-1][2]

      if not abs(c-p) < y_rng:
         g_id += 1
         groups.append([])

      groups[g_id].append(letters[ii])

   better = []

   for g in groups:
      # It isnt a group
      if len(g) < 3:
         continue 

      g = sorted(g, key=lambda x: x[1])

      better.append(g)
   return better

def Clean_Better(mask):
   h,w = mask.shape[:2]

   new = mask.copy()

   rng = 1

   for x in range(0,w-rng):
      for y in range(0,h-rng):
         v1 = 0
         v2 = 0

         v1 += mask[y,x]
         v1 += mask[y+1,x+1]
         v2 += mask[y+1,x]
         v2 += mask[y,x+1] 

         if (v1 == 0 and v2 == 510) or (v2 == 0 and v1 == 510):
            new[y,x] = 0
            new[y+1,x] = 0
            new[y,x+1] = 0
            new[y+1,x+1] = 0

   return new


def Find_Missing(letters,image,mask):
   letters = sorted(letters, key=lambda x: x[1])

   x1,y1,w1,h1 = cv2.boundingRect(letters[0][3])
   x2,y2,w2,h2 = cv2.boundingRect(letters[1][3])

   buff = 4

   x = x1 - (x2-x1)- buff
   y = y1 - (y2-y1)- buff


   w = int((w1+w2)/2) + 3*buff
   h = int((h1+h2)/2) + 3*buff

   crop = image[y:y+h,x:x+w]
   gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
   mask2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,1)

   crop2 = mask[y:y+h,x:x+w]

   better = cv2.bitwise_and(mask2, crop2)

   better = Clean_Better(better)

   Trim_Edges(better, color=0, width=1)

   res = cv2.findContours(better, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   # cnts = Get_Contours(better)
   cnts = sorted(cnts,key=lambda x: cv2.contourArea(x))

   c = cnts[-1]
   better[np.where(better != 0)] = 0
   cv2.drawContours(better,[c],0,255,-1)
   x3,y3,w3,h3 = cv2.boundingRect(c)
   better2 = better[y3:y3+h3,x3:x3+w3]
   better2 = cv2.resize(better2,(50,50))

   guess = Test_Number(better2)

   c+=(x,y)

   # cv2.imshow("mask2",mask2)
   # # cv2.imshow("better3",better3)
   # cv2.imshow("better",better)
   # cv2.imshow("better2",better2)
   # cv2.imshow("crop2",crop2)
   # cv2.waitKey(0)
   return [guess,x+x3,y+y3,c]

   # h = h1
   # x = x1 - int(max(w1,w2) * 1.2)
   # y = y1 - (y2-y1)
   # w = max(w1,w2)

   cv2.rectangle(image, (x,y),(x+w,y+h), (255,0,255), 1)

   crop = mask[y:y+h,x:x+w]
   cnts = Get_Contours(crop)

   # c = cnts[0]
   # x3,y3,w3,h3 = cv2.boundingRect(c)
   # crop = crop[y3:y3+h3,x3:x3+w3]
   # crop = cv2.resize(crop,(50,50))
   # crop[np.where(crop != 0)] = 255

   # guess = Test_Number(crop)

   # c+=(x,y)

def Find_Gray(image):

   image = cv2.bilateralFilter(image, 15,25,25)

   h,w = image.shape[:2]

   mask = np.zeros(image.shape[:2],dtype="uint8")

   min_value = 100
   var = 30

   for y in range(h):
      row = image[y]
      for x in range(w):
         avg = sum(row[x]) / 3

         # Not dark enough
         if avg < min_value:
            continue

         # Is the variance to big?
         valid = True
         for ii in range(3):
            if abs(image[y,x,ii] - avg) > var:
               valid = False

         if not valid:
            continue

         if avg > 100:
            mask[y,x] = 255


   # mask = cv2.medianBlur(mask,3)

   return mask

def Main(files):

   answers = Setup_Verifier("Directional")


   for f in files:

      fname = os.path.basename(f).split(".")[0]

      image = cv2.imread(f)

      mask = Get_Mask(image)

      mask = Clean(mask)

      better = Remove_Negatives(mask)

      letters = Find_Numbers(better)

      groups = Group_Letters(letters)

      ii = 0
      print("="*20)
      for g in groups:

         if len(g) == 3:
            guess = Find_Missing(g, image, mask)
            g.insert(0,guess)

         for c in g:
            x,y,w,h = cv2.boundingRect(c[3])  
            buf = 0
            cv2.rectangle(image, (x-buf,y-buf),(x+w+buf,y+h+buf), (0,0,255), 1)

         number = "".join([ii[0] for ii in g])
         answer = answers[fname][ii]
         print(answer,number)

         ii += 1 

      cv2.imshow("better",better)
      cv2.imshow("mask",mask)
      cv2.imshow("image",image)
      cv2.waitKey(0)

if __name__ == "__main__":
   files = glob.glob("DirectionalSignage/*")

   for f in files:
      image = cv2.imread(f)

      gray = Find_Gray(image)

      cv2.imshow("image",image)
      cv2.imshow("gray",gray)
      cv2.waitKey(0)

   # Main(files)

