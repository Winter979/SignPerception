import cv2
import glob
import imutils

import numpy as np

from Tools import *

def Contour_Mask(orig_mask):
   
   mask = orig_mask.copy()
   for ii in range(5):
      mask = cv2.medianBlur(mask, 51)
   
   # mask = np.ones(orig_mask.shape[:2],dtype="uint8") * 255

   # cnts = cv2.findContours(~orig_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   # cnts = imutils.grab_contours(cnts) 

   # cv2.drawContours(mask, cnts, -1, 0, -1)

   return mask

def Find_Possible(mask):
   cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   cnts = imutils.grab_contours(cnts)  
   
   new = np.ones(mask.shape[:2],dtype="uint8") * 255

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      area = w*h

      rect = cv2.minAreaRect(c)
      (x1,x2),(w1,h1),angle = rect

      angle = abs(angle)

      if 400 < area < 4000:
         # print(rect)
         ratio = h/w
         if 1 < ratio < 3:
            if angle < 30 or angle > 60:
               cv2.drawContours(new,[c],-1,0,-1)

   return new

def Group_Letters(image, mask):

   cnts = cv2.findContours(~mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
   cnts = imutils.grab_contours(cnts)

   count = len(cnts)

   # Not enough
   if count < 3:
      return 

   better = []

   miny = image.shape[0]
   maxy = 0

   # Remove the small ones
   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      ratio = h/w
      area = w*h

      # rect = cv2.minAreaRect(c)
      # (x1,x2),(w1,h1),angle = rect

      # angle = abs(angle)

      if 1 < ratio < 3 and area > 300:
         better.append(c)

         maxy = y if y > maxy else maxy
         miny = y if y < miny else miny


   by_y = sorted(better, key=lambda x: cv2.boundingRect(x)[1])

   rng = 50

   groups = []

   for ii in range(miny, maxy, rng):

      lower = ii
      upper = ii + rng

      group = []
      temp = by_y.copy()

      for jj in range(len(temp)):
         c = temp[jj]
         y = cv2.boundingRect(c)[1]

         # Its within range
         if lower <= y <= upper:
            group.append(c)
         # Will no longe fit in any of them
         elif y < lower:
            by_y.remove(c)
         # No more are within range
         elif y > upper:
            break
            
      # Expecting more than 2 numbers
      if len(group) > 2:
         groups.append(group)


   if len(groups) == 1:
      letters = groups[0]
      # cv2.drawContours(image,letters,-1,(0,255,0),2)

      x1,y1,w,h = cv2.boundingRect(letters[0])

      x2 = x1 + w
      y2 = y1 + h

      for l in letters:
         x,y,w,h = cv2.boundingRect(l)
         
         if x < x1:
            x1 = x
         if y < y1:
            y1 = y

         x += w
         y += h

         if x > x2:
            x2 = x
         if y > y2:
            y2 = y

      x1 -= 5
      y1 -= 5
      x2 += 5
      y2 += 5

      cv2.rectangle(image, (x1,y1),(x2,y2), (0,255,0), 2)

def main(files):
   for f in files:
      image = cv2.imread(f)

      # image = cv2.medianBlur(image, 5)
      # _,gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

      _,white = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
      _,dark  = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)

      temp = np.ones(image.shape[:2],dtype="uint8") * 255
      temp[np.where((white == [0,0,0]).all(axis=2))] = 0

      dark_mask = np.ones(image.shape[:2],dtype="uint8") * 255
      dark_mask[np.where((dark == [0,0,0]).all(axis=2))] = 0
      
      Trim_Edges(temp, color=0, width=5)
      Trim_Edges(dark_mask, color=255, width=5)
      
      poss = Find_Possible(temp)

      mask = Contour_Mask(dark_mask)

      new = poss + mask

      new[np.where(new != 0)] = 255

      Group_Letters(image, new)

      cv2.imshow("image",image)
      # cv2.imshow("dark_mask",dark_mask)
      # cv2.imshow("poss",poss)
      # cv2.imshow("mask",mask)
      cv2.imshow("new",new)
      
      cv2.waitKey(0)
      
if __name__ == "__main__":
   files = glob.glob("DirectionalSignage/*")
   main(files)