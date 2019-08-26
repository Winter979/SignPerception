import cv2
import numpy as np
import glob
import imutils

class Item:
   def __init__(self,x,y,w,h,cnt):
      self.x = x
      self.y = y
      self.w = w
      self.h = h
      self.cnt = cnt


def Get_Files(part=1, test=False):

   if part == 1:
      if test:
         files = ["./BuildingSignage/BS08.jpg"]
      else:
         files = glob.glob("BuildingSignage/*")
   elif part == 2:
      if test:
         files = ["./BuildingSignage/DS02.jpg"]
      else:
         files = glob.glob("DirectionalSignage/*")

   return files

def Cvt_All(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

   return gray, hsv

def Filter_White(image, color="hsv"):

   if color == "hsv":
      lower_white = np.array([0,0,0])
      upper_white = np.array([360,100,120])
      mask_white = cv2.inRange(image, lower_white, upper_white)
   elif color == "gray":
      lower_white = np.array([180])
      upper_white = np.array([255])
      mask_white = cv2.inRange(image, lower_white, upper_white)

   return mask_white

def Filter_Dark(image, color="hsv"):

   if color == "hsv":
      lower_gray = np.array([0,0,0])
      upper_gray = np.array([255,180,120])
      mask_gray = cv2.inRange(image, lower_gray, upper_gray)
   elif color == "rgb":
      lower_gray = np.array([0,0,0])
      upper_gray = np.array([100,100,100])
      mask_gray = cv2.inRange(image, lower_gray, upper_gray)

   return mask_gray

def Refine_It(image):

   gray, hsv = Cvt_All(image)
   mask = Filter_Dark(hsv)

   h,w,_ = image.shape
   area = w*h

   items = []
   
   canny = cv2.Canny(mask, 200,220)

   cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)

   # cv2.drawContours(gray, cnts, -1, (0,255,0), 2)

   ii = 0
   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)

      item_area = w*h

      if item_area >= area * 0.03:
         ratio = w/h

         if 0.2 < ratio < 0.8:
            crop = mask[y:y+h,x:x+w]

            item = Item(x,y,w,h,c)

            items.append(item)

            Quadrant_Image(crop)

            cv2.imshow("crop{}".format(ii), crop)
            # cv2.imwrite("characters/crop{}.jpg".format(ii), crop)
            ii += 1


   # cv2.imshow("gray", mask)

   cv2.imshow("item", image)
   # cv2.imshow("cnts", cnts)

def Quadrant_Image(image):
   h,w = image.shape
   x_max = 5
   y_max = 7

   cell_width = w//x_max
   cell_height = h//y_max

   quads = []

   print("="*30)

   for ii in range(x_max):
      for jj in range(y_max):
         x = cell_width * ii
         y = cell_height * jj

         crop = image[y:y+cell_height,x:x+cell_width]

         ratio =Get_Ratio(crop)

         quads.append(ratio)

   print(quads)

def Get_Ratio(image):

   cells = 0
   count = 0

   h,w = image.shape

   for x in range(w):
      for y in range(h):
         count += 1
         if image[y,x] == 255:
            cells += 1
      
   ratio = int((cells / count) * 1000) / 1000

   return ratio

def Clean_Crop(image):

   gray, _ = Cvt_All(image)

   cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

   for c in cnts:
      approx = cv2.approxPolyDP(c, 0.01*cv2.arcLength(c, True),True)

      if len(approx) > 3:
         cv2.drawContours(image, [c],0,(0,255,0),-1)



   cv2.imshow("item",image)