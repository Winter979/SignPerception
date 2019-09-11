import cv2
import numpy as np

from Tools import *

def Find_Arrows(image):
 
   image = cv2.bilateralFilter(image,15,75,75)

   mser = cv2.MSER_create()

   #Convert to gray scale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,3,1)

   regions, _ = mser.detectRegions(gray)
   hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

   mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)
   mask1 = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

   arrows = []
   poss = []

   for c in hulls:
      x,y,w,h = cv2.boundingRect(c)

      area = w*h

      if 100 < area < 1000:
         rect = cv2.minAreaRect(c)
         (x1,x2),(w1,h1),angle = rect

         ratio = h/w
         ratio2 = w1/h1
         if Is_Arrow(ratio, ratio2, angle):
            # arrows.append([c,x,y])
            cv2.drawContours(mask, [c],0,255,-1)
         else:
            # poss.append([c,x,y])
            cv2.drawContours(mask1, [c],0,125,-1)

   # Merge the arrow contours
   res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      arrows.append([c,x,y])
   arrows = sorted(arrows, key=lambda x: x[1])

   # Remove X outliers
   old = arrows
   arrows = []

   mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

   for ii in range(len(old)):
      x = old[ii][1]
      p = old[ii-1][1]
      n = old[ii+1][1] if ii != len(old)-1 else 0

      diff = min(abs(x-p),abs(x-n))

      if diff < 10:
         arrows.append(old[ii])
         cv2.drawContours(mask, [old[ii][0]],0,255,-1)
      else:
         # cv2.drawContours(mask, [old[ii][0]],0,125,-1)
         pass

   # Filter possibilities
   res = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res    if len(res) == 2 else res[1:3]


   for c in cnts:

      if Close_Enough(arrows, c):
         cv2.drawContours(mask, [c],0,255,-1)
      else:
         pass
         # cv2.drawContours(mask, [c],0,125,0)

      # x,y,w,h = cv2.boundingRect(c)
      # poss.append([c,x,y])

   return mask

def Close_Enough(arrows, item):
   x,y,w,h = cv2.boundingRect(item)


   for c in arrows:
      x1,y1,w1,h1 = cv2.boundingRect(c[0])

      # Its behind it
      if x > x1-(w/2):
         continue

      dx = x1-x

      if 10 < dx < 4.5*w1:
         dy = y1-y
         if -5 < dy < 10:
            if h > h1*1.1:
               if w < w1 * 1.3:
                  return True;
   # print(w1)

   return False

def Is_Arrow(r1,r2,a):
   if 0.9<r1<1.1:
      if 0.9<r2<1.1:
         if 40 < abs(a) < 50:
            return True

   return False


def Find_Negatives(image):
   
   blur = cv2.bilateralFilter(image, 75, 50,50)

   gray,hsv = Cvt_All(blur)

   mask = np.zeros(image.shape[:2],dtype="uint8")

   colors = ["green", "red1","red2","yellow","sky","brown"]
   for c in colors:
      temp = Create_Mask(hsv, c)
      mask = cv2.bitwise_or(mask,temp)

   mask = cv2.medianBlur(mask, 3)

   return mask

def Magic(image, mask):
   res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]

   groups = []

   # Sort by Y
   cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])

   sort = cnts
   cnts = []

   groups = []
   new = []
   new.append(sort[0])

   pY = cv2.boundingRect(sort[0])[1]

   for ii in range(1,len(sort)):
      y = cv2.boundingRect(sort[ii])[1]
      if -10 < y-pY < 10:
         new.append(sort[ii])
      else:
         groups.append(new)
         new = []
         new.append(sort[ii])
      pY = y

   groups.append(new)


   for g in groups:
      while len(g) > 4:
         g.pop(0)
         
      g = sorted(g, key=lambda x: cv2.boundingRect(x)[0])

      if len(g) == 4:
         for c in g:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
      elif len(g) == 3:
         for c in g:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
         x1,y1,w1,h1 = cv2.boundingRect(g[0])
         x2,y2,w2,h2 = cv2.boundingRect(g[1])

         x = x1 - (x2-x1)
         y = y1 - (y2-y1)

         cv2.rectangle(image,(x,y),(x+int((w1+w2)/2),y+int((h1+h2)/2)),(0,255,255),1)

   # for ii in range(0,len(cnts),3):
   #    new = [cnts[ii],cnts[ii+1],cnts[ii+2]]

   #    new = sorted(new, key=lambda x: cv2.boundingRect(x)[0])

   #    while len(new) > 4:
   #       new.pop(0)

   #    if len(new) == 4:
   #       for c in new:
   #          x,y,w,h = cv2.boundingRect(c)
   #          cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
   #    elif len(new) == 3:
   #       x1,y1,w1,h1 = cv2.boundingRect(new[0])
   #       x2,y2,w2,h2 = cv2.boundingRect(new[1])

   #       x = x1 - (x2-x1)
   #       y = y1 - (y2-y1)

   #       cv2.rectangle(image,(x,y),(x+w1,y+h1),(0,255,255),1)

   #    groups.append(new)
   
def DefsNot(image):
   # mask = np.zeros(image.shape[:2],dtype="uint8")

   # blur = cv2.bilateralFilter(image, 75, 50,50)

   gray,hsv = Cvt_All(image)

   # colors = ["green", "red1","red2","yellow","brown","sky","dark"]
   # for c in colors:
   #    temp = Create_Mask(hsv, c)
   #    mask = cv2.bitwise_or(mask,temp)

   lower = np.array([0,80,0])
   upper = np.array([255,255,255])
   ok = cv2.inRange(hsv, lower, upper)

   # sky = Get_The_Sky(image)
   # mask = cv2.bitwise_or(mask,sky)

   return ok


def Get_Borders(image):
   gray,_ = Cvt_All(image)

   # n0 = DefsNot(image)

   idk = np.ones(image.shape[:2],dtype="uint8") * 255

   ret,temp = cv2.threshold(image,100,255,cv2.THRESH_BINARY)

   th1 = np.zeros(image.shape[:2],dtype="uint8")
   th1[np.where((temp == [255,255,255]).all(axis=2))] = 255

   th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,7,5)

   th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,25,0)

   th4 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,55,0)

   th5 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,99,5)

   idk = cv2.bitwise_and(idk, th1)
   idk = cv2.bitwise_and(idk, th2)
   idk = cv2.bitwise_and(idk, th3)
   idk = cv2.bitwise_and(idk, th4)
   idk = cv2.bitwise_and(idk, th5)
   # idk = cv2.bitwise_and(idk, ~n0)

   return idk

def main(files):
   for f in files:
      image = cv2.imread(f)

      temp = image.copy()


      # idk = Get_Borders(image)
      # temp = cv2.bitwise_and(temp,temp, mask=idk)
      # idk = DefsNot(image)
      # temp = cv2.bitwise_and(temp,temp, mask=~idk)

      # canny = cv2.Canny(temp, 200, 220)

      new = Find_Arrows(image)

      Magic(image, new)

      cv2.imshow("image",image)
      # cv2.imshow("canny",canny)
      cv2.imshow("new",new)
      # cv2.imshow("idk",idk)
      cv2.waitKey(0)
      
