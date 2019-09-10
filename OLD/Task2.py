import cv2
import numpy as np
from Tools import *

import Testing as t

def Magic(image):
   mser = cv2.MSER_create()

   #Convert to gray scale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   vis = image.copy()

   #detect regions in gray scale image
   regions, _ = mser.detectRegions(gray)

   hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

   mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

   better = []
   

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
                  # xs.append([x,c])
                  # better.append(c)
                  cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)
   
   res = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
   
   xs = []
   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      xs.append([x,c])

   xs = sorted(xs, key=lambda x: x[0])

   better = []

   maxX = 0

   print("=============")
   for ii in range(len(xs)):
      x = xs[ii][0]
      p = xs[ii-1][0]
      n = xs[ii+1][0] if ii != len(xs)-1 else 0

      diff = min(abs(x-p),abs(x-n))
      if diff < 5:
         maxX = x if x > maxX else maxX
         better.append(xs[ii][1])

   # ys = []
   # ws = []
   # xs = []
   for c in better:
      x,y,w,h = cv2.boundingRect(c)

      x1 = x - int(w*1.4)
      x2 = x - int(w*.5)

      y2 = y - int(h*.2)
      y1 = y + int(h*1.2)

      cv2.rectangle(image, (x1,y1),(x2,y2), (0,255,0), 2)

      x1 = x - int(w*2.6)
      x2 = x - int(w*1.7)

      y2 = y - int(h*.2)
      y1 = y + int(h*1.2)

      cv2.rectangle(image, (x1,y1),(x2,y2), (0,255,0), 2)

      x1 = x - int(w*3.7)
      x2 = x - int(w*2.8)

      y2 = y - int(h*.2)
      y1 = y + int(h*1.2)

      cv2.rectangle(image, (x1,y1),(x2,y2), (0,255,0), 2)

      # ws.append(w)
      # xs.append(x)
      # ys.append(y)

   # minX = min(xs)
   # maxX = max(xs)

   # minY = min(ys)
   # maxY = max(ys)
   
   # width = int(np.mean(ws))

   # y2 = maxY + int(width * 1.5)
   # y1 = minY - int(width / 2)

   # x1 = minX - int(3.8 * width)
   # x2 = maxX + int(1.2 * width)

   # cropped = image[y1:y2,x1:x2]

   # cv2.drawContours(mask, better, -1, 255, -1)
   # cv2.imshow('mask', mask)

   cv2.imshow('cropped', image)
   # cv2.imshow("text only", text_only)
   # cv2.waitKey(0)

   return mask

def Straight_Lines(image):
   
   image = image.copy()
   
   canny = cv2.Canny(image, 10,200)
   lines = cv2.HoughLines(canny,1,np.pi/180,230)

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

      cv2.line(mask,(x1,y1),(x2,y2),0,2)

   return mask

def Magic2(image):
   
   # blur = cv2.GaussianBlur(image, (5,5),0)
   # blur = cv2.bilateralFilter(image, 55,75,75)

   gray, hsv = Cvt_All(image)
   canny = cv2.Canny(image,150,200)

   kernel = np.ones((25,15),np.uint8)
   closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

   res = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   mask = np.zeros(image.shape[:2],dtype="uint8")

   iH,iW = image.shape[:2]

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)

      if w*h < 100:
         continue

      if w/iW > 0.3 or w/iW < 0.1:
         continue

      cv2.drawContours(mask,[c],0,255,1)

   cv2.imshow("closing",closing)

      
   return mask
   # Trim_Edges(canny)

   # kernel = np.ones((15,15),np.uint8)
   # temp = cv2.morphologyEx(~canny, cv2.MORPH_OPEN, kernel)

   # kernel = np.ones((5,5),np.uint8)
   # gradient = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, kernel)

   # kernel = np.ones((15,15),np.uint8)
   # closing = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

   # mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

   # res = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   # cnts,_ = res if len(res) == 2 else res[1:3]

   # xs = []
   # better = []

   # for c in cnts:
   #    x,y,w,h = cv2.boundingRect(c)

   #    if w*h > 1000:
   #       xs.append([x,c])
   #       # cv2.drawContours(mask,[c],0,255,-1)

   # xs = sorted(xs, key=lambda x: x[0])

   # for ii in range(len(xs)):
   #    x = xs[ii][0]
   #    p = xs[ii-1][0]
   #    n = xs[ii+1][0] if ii != len(xs)-1 else 0

   #    diff = min(abs(x-p),abs(x-n))
   #    if diff < 5:
   #       # maxX = x if x > maxX else maxX
   #       better.append(xs[ii][1])

   # cv2.drawContours(mask,better,-1,255,-1)

   # cv2.imshow("closing",closing)

   # res = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
   # cnts,_ = res if len(res) == 2 else res[1:3]

   # iH,iW = image.shape[:2]
   
   # mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

   # better =[]
   # xs = []

   # for c in cnts:
   #    x,y,w,h = cv2.boundingRect(c)

   #    if 4000 < w*h:
   #       if 0.1 < w/iW < 0.2:
   #          if h/iH > 0.01:
   #             xs.append([x,c])
   #             print(x)
   #             # cv2.drawContours(mask,[c],0,255,-1)

   # xs = sorted(xs, key=lambda x: x[0])

   # for ii in range(len(xs)):
   #    x = xs[ii][0]
   #    p = xs[ii-1][0]
   #    n = xs[ii+1][0] if ii != len(xs)-1 else 0

   #    diff = min(abs(x-p),abs(x-n))
   #    if diff < 5:
   #       # maxX = x if x > maxX else maxX
   #       better.append(xs[ii][1])
   
   # cv2.drawContours(mask,better,-1,255,-1)

   # cv2.imshow("closing",closing)
   # cv2.imshow("mask",mask)

def DefsNot(image):
   mask = np.zeros(image.shape[:2],dtype="uint8")

   blur = cv2.bilateralFilter(image, 75, 50,50)

   gray,hsv = Cvt_All(blur)

   colors = ["green", "red1","red2","yellow","brown","shadow"]
   for c in colors:
      temp = Create_Mask(hsv, c)
      mask = cv2.bitwise_or(mask,temp)

   sky = Get_The_Sky(image)
   mask = cv2.bitwise_or(mask,sky)

   return mask

def IDK(image):

   n0 = DefsNot(image)

   gray,hsv = Cvt_All(image)

   ret,temp = cv2.threshold(image,100,255,cv2.THRESH_BINARY)

   th1 = np.zeros(image.shape[:2],dtype="uint8")
   th1[np.where((temp == [255,255,255]).all(axis=2))] = 255

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,0)

   th4 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,25,0)

   idk = np.ones(image.shape[:2],dtype="uint8") * 255

   idk = cv2.bitwise_and(idk, th1)
   idk = cv2.bitwise_and(idk, th3)
   idk = cv2.bitwise_and(idk, th4)
   idk = cv2.bitwise_and(idk, ~n0)

   # idk = th1 - ~th3 - ~th4 - sky

   Trim_Edges(idk, color=0)

   mask = np.zeros(image.shape[:2],dtype="uint8")

   res = cv2.findContours(idk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      if 100 < w*h < 800:
         # if angle > 20 or angle > 70:
         cv2.drawContours(mask, [c],0,255,-1)

   mask = cv2.medianBlur(mask, 3)

   old = mask

   res = cv2.findContours(old, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   mask = np.zeros(image.shape[:2],dtype="uint8")

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)

      rect = cv2.minAreaRect(c)
      (x1,x2),(w1,h1),angle = rect

      angle = abs(angle)

      if 150 < w*h < 800:
         ratio = h/w
         ratio2 = h1/w1
         if 1.3 < ratio < 4 and not (30 < angle < 60):

            crop = old[y:y+h,x:x+w]

            try:
               magic = Get_Ratio(crop)
               letter = Guess_Letter(magic)
               print("Guessed",letter)
            except IndexError:
               print("Nothing found")

            cv2.imshow("crop",crop)
            cv2.waitKey(0)
            cv2.drawContours(mask, [c],0,255,-1)
         elif 40 < angle < 50:
            if 0.9 < w1/h1 < 1.1:
               pass
               cv2.drawContours(mask, [c],0,255,-1)
         else:
            cv2.drawContours(mask, [c],0,125,-1)

   return mask

def Try_Again(image):

   image = cv2.bilateralFilter(image, 55, 75,75)

   gray,hsv = Cvt_All(image)



   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,95,1)


   return th3

def main(files):
   for f in files:
      image = cv2.imread(f)

      new = Try_Again(image)
      # new = DefsNot(image)

      # new = Straight_Lines(image)

      # canny = cv2.Canny(image, 30,100)

      # blur = cv2.bilateralFilter(image, 55,75,75)
      # gray, hsv = Cvt_All(blur)

      # _,th1 = cv2.threshold(gray, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      # new = Magic3(image)
      # new = Magic(blur)

      # temp = Straight_Lines(new)

      # temp = image.copy()
      # temp[np.where()]

      cv2.imshow("image",image)
      cv2.imshow("new",new)
      # cv2.imshow("temp",temp)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

      # _,temp = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)

      # _,dark = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

      # mask = np.ones(image.shape[:2]) * 255
      # mask[np.where((dark == [0,0,0]).all(axis=2))] = 0
      # mask[np.where((dark == [255,0,0]).all(axis=2))] = 0
      # Remove_Stuff(blur)

      # new = Magic(image)

      # cv2.imshow("image",image)
      # cv2.imshow("new",new)
      # cv2.imshow("mask",mask)
      # cv2.imshow("magic",mask + new)
      # cv2.waitKey(0)

      # Magic(image)


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