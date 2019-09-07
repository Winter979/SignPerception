import cv2
import numpy as np
from Tools import *

def Magic(image):
   mser = cv2.MSER_create(_edge_blur_size=10)

   #Convert to gray scale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   vis = image.copy()

   #detect regions in gray scale image
   regions, _ = mser.detectRegions(gray)

   hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]


   mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

   for c in hulls:
      x,y,w,h = cv2.boundingRect(c)

      area = w*h


      if 100 < area < 700:
         rect = cv2.minAreaRect(c)
         (x1,x2),(w1,h1),angle = rect

         ratio = h/w
         ratio2 = w1/h1

         cv2.polylines(vis, [c], 2, (0, 255, 0))
         if 0.9<ratio<1.1:
            if 0.9<ratio2<1.1:
               if 35 < abs(angle) < 55:
                  cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)


   #this is used to find only text regions, remaining are ignored
   # text_only = cv2.bitwise_and(image, image, mask=mask)

   
   cv2.imshow('mask', mask)
   cv2.imshow('image', vis)
   # cv2.imshow("text only", text_only)
   cv2.waitKey(0)



def Straight_Lines(image):

   image = image.copy()

   canny = cv2.Canny(image, 150,200)
   lines = cv2.HoughLines(canny,1,np.pi/180,150)

   print("="*20)
   for line in lines:
      rho,theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)

      # print(a,b)

      if b > 0.05:
         continue

      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))

      print(x1,y1,x2,y2)



      cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

   return image

def Remove_Stuff(image):
   
   gray, hsv = Cvt_All(image)
   _,mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

   new = np.ones(image.shape[:2],dtype="uint8") * 0

   new[np.where((mask == [255,255,255]).all(axis=2))] = 255

   colors = ["green","brown","red1","red2","yellow"]
   for c in colors:
      mask = Create_Mask(hsv, c)
      new[np.where(mask == 255)] = 255


   cv2.imshow("image",image)
   cv2.imshow("new",new)
   cv2.waitKey(0)

def main(files):
   for f in files:
      image = cv2.imread(f)

      blur = cv2.bilateralFilter(image, 55,75,75)
      gray, hsv = Cvt_All(blur)

      # Remove_Stuff(blur)

      Magic(image)

      # cv2.imshow("image",image)
      # cv2.waitKey(0)

      # Magic(image)

      # _,mask = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)

      # dark = np.ones(mask.shape[:2],dtype="uint8") * 255
      # dark[np.where((mask == [0,0,0]).all(axis=2))] = 0
      # dark[np.where((mask == [255,0,0]).all(axis=2))] = 0

      # _,mask = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)

      # light = np.ones(mask.shape[:2],dtype="uint8") * 0
      # light[np.where((mask == [255,255,255]).all(axis=2))] = 255

      # new = light + dark

      # colors = ["green","brown","red1","red2","yellow","white"]
      # for c in colors:
      #    mask = Create_Mask(hsv, c)
      #    new[np.where(mask == 255)] = 255
      

      # cv2.imshow("image",image)
      # cv2.imshow("dark",dark)
      # cv2.imshow("light",light)
      # cv2.imshow("new",new)
      
      
      # cv2.waitKey(0)