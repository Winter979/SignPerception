import cv2
import glob
import os
import sys
import numpy as np

from Settings import Settings as s

from Tools import *

def Contour_Mask(orig_mask):
   
   mask = orig_mask.copy()
   for ii in range(5):
      mask = cv2.medianBlur(mask, 51)

   return mask

def Find_Possible(mask):
   res = cv2.findContours(~mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

      if 200 < area < 4000 and area2 > 100:
         ratio = h/w
         if 1 < ratio < 4:
            cv2.drawContours(new,[c],-1,0,-1)

   return new

def Possible_Letters(mask):
   res = cv2.findContours(~mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   count = len(cnts)

   # Not enough
   if count < 3:
      return None

   better = []

   # Remove the small ones
   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      ratio = h/w
      area = w*h
      area2 = cv2.contourArea(c)

      if 1 < ratio < 3 and area > 300 and area2 > 100:
         better.append(c)


   return better

def Group_By_Y(cnts):
   by_y = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])

   miny = cv2.boundingRect(by_y[0])[1]
   maxy = cv2.boundingRect(by_y[-1])[1]

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
         # Will no longe fit in any of them so remove it (makes looping faster..theoretically)
         elif y < lower:
            by_y.remove(c)
         # No more are within range
         elif y > upper:
            break
            
      # Expecting more than 2 numbers
      if len(group) > 2:
         groups.append(group)

   if len(groups) == 0:
      return None

   return groups

def Group_By_X(y_groups):
   filtered = []

   # Remove large x gaps
   for g in y_groups:
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

   sorts = []
   for temp in filtered:
      sorts.append(sorted(temp, key=lambda x: cv2.boundingRect(x)[0]))

   return sorts

def Letters_Bounding_Box(image, letters, show=False, draw=False):
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

   image_crop = image[y1:y2,x1:x2]

   if show:
      cv2.imshow("crop", image_crop)
   if draw:
      cv2.rectangle(image, (x1,y1),(x2,y2), (0,255,0), 2)

   return x1,y1,x2,y2

def Extract_Letters(image, letters, draw=False, show=False):
   ii = 0

   # Reapply threshold to the numbers
   _,mask = cv2.threshold(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)

   guess = []

   for l in letters:
      x,y,w,h = cv2.boundingRect(l)

      mask_crop = mask[y:y+h,x:x+w]
      image_crop = image[y:y+h,x:x+w]

      if show:
         cv2.imshow("letter-{}".format(ii), image_crop)
      if draw:
         cv2.rectangle(image, (x-2,y-2),(x+w+2,y+h+2), (0,0,255), 1)

      ratio = Get_Ratio(mask_crop)
      letter = Guess_Letter(ratio)

      guess.append(letter)

      ii += 1

   return guess

def Get_White_Mask(image):
   _,mask = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY_INV)
   
   # Convert mask to white & black
   binary = np.ones(image.shape[:2],dtype="uint8") * 255
   binary[np.where((mask == [0,0,0]).all(axis=2))] = 0

   # Trim the edges to assist in contour functions
   Trim_Edges(binary, color=255, width=5)

   # Remove any light areas that are gold (Signs arent gold)
   gold_mask = Gold_Mask(image)
   binary[np.where(gold_mask == 255)] = 255

   # Only keep areas that could be a number
   binary = Find_Possible(binary)

   return binary

def Get_Dark_Mask(image):
   _,mask = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)
   
   # Convert mask to white & black
   binary = np.ones(image.shape[:2],dtype="uint8") * 255
   binary[np.where((mask == [0,0,0]).all(axis=2))] = 0

   # Trim the edges to assist in contour functions
   Trim_Edges(binary, color=255, width=5)

   return binary

def Setup_Verifier():
   data = {}
   with open("./Answers.txt") as f:
      lines = f.read().splitlines()
      for line in lines:
         s = line.split(":")
         name = s[0]
         answer = s[1]
         data[name] = answer

   return data

def main(files):

   answers = Setup_Verifier()

   tests = 0
   passed = 0

   try:

      for f in files:
         tests += 1
         # The name of the file (Used for the final printing)
         fname = os.path.basename(f).split(".")[0]

         try:

            image = cv2.imread(f)      

            white = Get_White_Mask(image)
            dark = Get_Dark_Mask(image)

            filled_dark = Contour_Mask(dark)

            # Create the final mask + cleanup
            mask = white + filled_dark
            mask[np.where(mask != 0)] = 255

            # letter_groups = Group_Letters(image, mask)
            
            cv2.imshow("dark",dark)
            cv2.imshow("white",white)
            cv2.imshow("mask",mask)

            cnts = Possible_Letters(mask)
            if cnts == None:
               raise ValueError("No possibilities Found")

            y_groups = Group_By_Y(cnts)
            if y_groups == None:
               raise ValueError("No Letters found within a Y group")

            x_groups = Group_By_X(y_groups)
            if x_groups == None:
               raise ValueError("No Letters found within a X group")

            if len(x_groups) != 1:
               raise ValueError("To many possibilities. Refining time")
            

            Letters_Bounding_Box(image, x_groups[0], show=s.show, draw=s.draw)
            letters = Extract_Letters(image, x_groups[0], show=s.show)

            room = "".join(letters)
            
            print(room)
            cv2.imshow("image",image)
            if s.verbose:
               cv2.imshow("white",white)
               cv2.imshow("mask",mask)
               cv2.imshow("dark",dark)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue

            answer = answers[fname]

            if room == answer:
               passed += 1
               print("{}{} : {} == {}{}".format(Colrs.CYAN,fname, answer, room,Colrs.RESET))
            else:
               print("{}{} : {} != {}{}".format(Colrs.RED,fname, answer, room, Colrs.RESET))
            

         except ValueError as e:
            print("{}{} : {}{}".format(Colrs.RED,fname, e,Colrs.RESET))

         if s.show:
            cv2.imshow("image",image)
            if s.verbose:
               cv2.imshow("white",white)
               cv2.imshow("mask",mask)
               cv2.imshow("dark",dark)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
   except KeyboardInterrupt:
      tests -=1
      sys.stdout.write("\r")
      sys.stdout.flush()
      print("Tests aborted")

   print("RESULTS: {}/{}".format(passed, tests))

      # print(letters)

      # cv2.imshow("temp",temp)
      # cv2.imshow("white",white)
      # cv2.imshow("dark_mask",dark_mask)
      
      # cv2.waitKey(0)
      # cv2.destroyAllWindows()

if __name__ == "__main__":
   files = glob.glob("val_BuildingSignage/val03.jpg")
   main(files)