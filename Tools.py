import cv2
import glob
import numpy as np

class Colrs:
   RED      = "\033[1;31m"  
   BLUE     = "\033[1;34m"
   CYAN     = "\033[1;36m"
   GREEN    = "\033[0;32m"
   RESET    = "\033[0;0m"
   BOLD     = "\033[;1m"
   REVERSE  = "\033[;7m"

MASKS = {
   # BGR Masks
   "p_red"  : [np.array([  0,  0,150]) , np.array([ 30, 30,255])],
   "dark"   : [np.array([  0,  0,  0]) , np.array([125,125,125])],
   "p_green": [np.array([  0,200,  0]) , np.array([125,255,125])],
   # HSL Masks
   "pink"   : [np.array([100,  0,150]) , np.array([150, 30,200])],
   "white"  : [np.array([  0,  0,200]) , np.array([255, 50,255])],
   "red1"   : [np.array([  0, 50, 80]) , np.array([ 30,255,255])],
   "red2"   : [np.array([160, 50, 80]) , np.array([180,255,255])],
   "yellow" : [np.array([ 21, 60, 80]) , np.array([ 40,255,255])],
   "brown"  : [np.array([  5,100, 20]) , np.array([ 30,255,255])],
   "green"  : [np.array([ 41, 60, 64]) , np.array([ 90,255,255])],
   "shadow" : [np.array([  0,110,  0]) , np.array([ 60,255, 80])]
}

KNOWN = []
with open("./learn.txt") as f:
   lines = f.read().splitlines()
   for line in lines:
      res = int(line[0])
      ratios = [float(ii) for ii in line[2:].split(",")]
      KNOWN.append([res,ratios])

def Gold_Mask(image):
   gray,hsv = Cvt_All(image)

   lower = np.array([10,100,100])
   upper = np.array([80,255,255])
   inrange = cv2.inRange(hsv, lower, upper)

   inrange = cv2.medianBlur(inrange, 5)

   res = cv2.findContours(inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   cnts,_ = res if len(res) == 2 else res[1:3]
   
   for c in cnts:
      area = cv2.contourArea(c)

      if area < 1000:
         cv2.drawContours(inrange, [c], 0, 255, -1)
         
   inrange[np.where(inrange != 0)] = 255

   return inrange


def Create_Mask(image, color):

   lower, upper = MASKS[color]

   mask = cv2.inRange(image, lower, upper)
   # mask = cv2.medianBlur(mask, 7)

   return mask


def Trim_Edges(mask, color = 255, width=3):
   pack = mask.shape

   h,w = pack[:2]

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

def Guess_Letter(ratio):
   global KNOWN
   
   # print(KNOWN)

   guesses = []
   
   for ii in KNOWN:
      diff = Ratio_Diff(ratio, ii[1])
      # if diff < 5:
      guesses.append([diff, ii[0]])

   guesses = sorted(guesses,key= lambda x :x[0])

   letter = str(guesses[0][1])

   # Used for writing new learning data
   # with open("new.txt","a") as f:
   #    temp = ",".join("%.3f" % x for x in ratio)
   #    f.write("{}:{}\n".format(letter, temp))

   return letter


def Ratio_Diff(a,b):
   diff = 0

   for ii in range(len(a)):
      diff += abs(a[ii]-b[ii])

   return diff