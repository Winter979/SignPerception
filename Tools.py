import cv2
import numpy as np
import glob

import json

from Settings import Settings as s

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
   "gold"   : [np.array([ 10,100,100]) , np.array([ 80,255,255])],
   "white"  : [np.array([  0,  0,155]) , np.array([255, 100,255])],
   "red1"   : [np.array([  0, 80, 100]) , np.array([ 30,255,255])],
   "red2"   : [np.array([160, 80, 100]) , np.array([180,255,255])],
   "yellow" : [np.array([ 21, 60, 80]) , np.array([ 40,255,255])],
   "green"  : [np.array([ 41, 60, 64]) , np.array([ 90,255,255])],
   "brown"  : [np.array([  5,100, 20]) , np.array([ 30,255,255])],
   "shadow" : [np.array([  0,110,  0]) , np.array([ 60,255, 80])],
   "sky"    : [np.array([100, 60,200]) , np.array([140,255,255])],
   "dark"   : [np.array([  0, 60,200]) , np.array([255,255,255])],
   "red"    :[[np.array([  0, 80, 100]) , np.array([ 30,255,255])],
             [np.array([160, 80, 100]) , np.array([180,255,255])]],
}

KNOWN = []
with open("./learn.txt") as f:
   lines = f.read().splitlines()
   for line in lines:
      res = line[0]
      ratios = [float(ii) for ii in line[2:].split(",")]
      KNOWN.append([res,ratios])

def Gold_Mask(image):
   gray,hsv = Cvt_All(image)

   lower = np.array([10,100,100])
   upper = np.array([80,255,255])
   inrange = cv2.inRange(hsv, lower, upper)

   inrange = cv2.medianBlur(inrange, 5)

   # Trim_Edges(inrange)

   res = cv2.findContours(inrange, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   cnts,_ = res if len(res) == 2 else res[1:3]
   
   for c in cnts:
      area = cv2.contourArea(c)

      if area < 1000:
         cv2.drawContours(inrange, [c], 0, 255, -1)
         
   inrange[np.where(inrange != 0)] = 255

   return inrange


def Get_The_Sky(image):
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

   lower = np.array([100,60,200])
   upper = np.array([140,255,255])

   mask = cv2.inRange(hsv,lower,upper)
   return mask

def Create_Mask(image, color):


   if color == "red":
      lower, upper = MASKS[color][0]
      mask1 = cv2.inRange(image, lower, upper)

      lower, upper = MASKS[color][1]
      mask2 = cv2.inRange(image, lower, upper)

      mask = cv2.bitwise_or(mask1, mask2)
   else:
      lower, upper = MASKS[color]
      mask = cv2.inRange(image, lower, upper)


   # mask = cv2.medianBlur(mask, 7)

   Trim_Edges(mask, color=0)

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

def Get_Contours(mask):
   res = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   return cnts


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

   guesses = sorted(guesses, key=lambda x: x[1])

   if guesses[0][1] > 10000:
      return '?'
   else:
      return guesses[0][0]

def Create_Empty(image,color=0):
   new = np.zeros(image.shape[:2],dtype="uint8")
   
def Setup_Verifier(category):
   
   with open("./Answers.json") as f:
      data = json.load(f)   

   return data[category]