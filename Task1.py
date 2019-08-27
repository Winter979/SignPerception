import cv2
import glob
import imutils

import numpy as np

from Tools import *

# ANSWERS = []

def Contour_Mask(orig_mask):
   
   mask = orig_mask.copy()
   for ii in range(5):
      mask = cv2.medianBlur(mask, 51)

   return mask

def Find_Possible(mask):
   cnts = cv2.findContours(~mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
   cnts = imutils.grab_contours(cnts)  
   
   new = np.ones(mask.shape[:2],dtype="uint8") * 255

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      area = w*h
      
      area2 = cv2.contourArea(c)

      rect = cv2.minAreaRect(c)
      (x1,x2),(w1,h1),angle = rect

      angle = abs(angle)

      if 200 < area < 4000 and area2 > 100:
         ratio = h/w
         if 1 < ratio < 4:
            cv2.drawContours(new,[c],-1,0,-1)

   return new

def Group_Letters(image, mask):

   cnts = cv2.findContours(~mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
   cnts = imutils.grab_contours(cnts)

   count = len(cnts)

   # Not enough
   if count < 3:
      print("Not enough")
      return 

   better = []

   miny = image.shape[0]
   maxy = 0

   # Remove the small ones
   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      ratio = h/w
      area = w*h
      area2 = cv2.contourArea(c)

      if 1 < ratio < 3 and area > 300 and area2 > 100:
         better.append(c)

         maxy = y if y > maxy else maxy
         miny = y if y < miny else miny


   by_y = sorted(better, key=lambda x: cv2.boundingRect(x)[1])

   rng = 50

   groups = []

   for ii in range(miny, maxy+1, rng):

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

   if len(groups) == 0:
      print("nope")
      return

   filtered = []

   # Remove large x gaps
   for g in groups:
      xs = []
      for c in g:
         x = cv2.boundingRect(c)[0]
         xs.append([c,x])
      
      good = []
      for ii in range(len(xs)):
         x1 = xs[ii][1]
         ok = False 
         for jj in range(len(xs)):
            if ii != jj:
               x2 = xs[jj][1]
               if abs(x1-x2) < 60:
                  ok = True
                  break

         if ok:
            good.append(xs[ii][0])

      if len(good) == 3:
         filtered.append(good)

   if len(filtered) == 1:
      letters = filtered[0]
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

      # Add some padding to the crop
      x1 -= 10
      y1 -= 10
      x2 += 10
      y2 += 10

      # mask_crop = mask[y1:y2,x1:x2]

      Process_Letters(image, letters)

      image_crop = image[y1:y2,x1:x2]
      cv2.imshow("crop", image_crop)
      cv2.rectangle(image, (x1,y1),(x2,y2), (0,255,0), 2)

def Process_Letters(image, letters):

   gray,_ = Cvt_All(image)
   _,mask = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)

   letters = sorted(letters, key=lambda x: cv2.boundingRect(x)[0])

   ii = 0

   # global ANSWERS
   # expected = ANSWERS.pop(0)
   # print(expected)

   for l in letters:
      x,y,w,h = cv2.boundingRect(l)

      mask_crop = mask[y:y+h,x:x+w]
      image_crop = image[y:y+h,x:x+w]

      # cv2.rectangle(image, (x-2,y-2),(x+w+2,y+h+2), (0,0,255), 1)
      cv2.imshow("letter-{}".format(ii), image_crop)

      ratio = Get_Ratio(mask_crop)
      ii += 1

      # with open("answers.txt","a+") as f:
         # temp = ",".join("%.3f" % x for x in ratio)
         
         # f.write("{}:{}\n".format(expected.pop(0), temp))

def Get_Ratio(mask):
   
   h,w = mask.shape

   cuts = 5
   cell_width = w//cuts
   cell_height = h//cuts

   ratios = []

   for ii in range(cuts):
      for jj in range(cuts):
         cells = 0
         count = 0

         x = cell_width * ii
         y = cell_height * jj

         for x1 in range(x,x+cell_width):
            for y1 in range(y,y + cell_height):
               count += 1
               if mask[y1,x1] == 255:
                  cells += 1
         ratios.append(cells / count)

   return ratios

def main(files):

   # global ANSWERS

   # with open("./Excepted.txt") as f:
   #    lines = f.read().splitlines()
   #    for line in lines:
   #       sep = line.split(",")[1:]
   #       sep = [int(ii) for ii in sep]
   #       ANSWERS.append(sep)

   for f in files:
      image = cv2.imread(f)      

      _,white = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
      _,dark  = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)

      temp = np.ones(image.shape[:2],dtype="uint8") * 255
      temp[np.where((white == [0,0,0]).all(axis=2))] = 0

      dark_mask = np.ones(image.shape[:2],dtype="uint8") * 255
      dark_mask[np.where((dark == [0,0,0]).all(axis=2))] = 0
      
      Trim_Edges(temp, color=0, width=5)
      Trim_Edges(dark_mask, color=255, width=5)
      
      poss = Find_Possible(temp)
      gold_mask = Gold_Mask(image)
      poss[np.where(gold_mask == 255)] = 255

      mask = Contour_Mask(dark_mask)

      new = poss + mask

      new[np.where(new != 0)] = 255

      Group_Letters(image, new)

      cv2.imshow("image",image)
      # cv2.imshow("dark_mask",dark_mask)
      # cv2.imshow("poss",poss)
      # cv2.imshow("temp",dark_mask + new)
      # cv2.imshow("white",white)
      # cv2.imshow("new",new)
      
      cv2.waitKey(0)
      cv2.destroyAllWindows()

if __name__ == "__main__":
   files = glob.glob("DirectionalSignage/*")
   main(files)