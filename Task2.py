import cv2
import numpy as np
from Tools import *


def Filter_Image(image):
   canny = cv2.Canny(image,100,200)

   cv2.imshow("canny",canny)

   lines = cv2.HoughLinesP(image=canny,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=100,maxLineGap=20)

   for line in lines:
      x1,y1,x2,y2 = line[0]
      cv2.line(image, (x1,y1),(x2,y2), (0,255,0),2)

def Magic(image, mask):
   cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = cnts if len(cnts) == 2 else cnts[1:3]

   new = np.ones(mask.shape[:2],dtype="uint8") * 255

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)

      a2 = cv2.contourArea(c)

      area = w*h
      if 100 < area < 500:
         ratio = h/w
         if 0.8<ratio<4:
            cv2.drawContours(new,[c],-1,0,-1)

   cv2.imshow("new",new)


def main(files):
   for f in files:
      image = cv2.imread(f)

      gray, hsv = Cvt_All(image)      
      # equ = cv2.equalizeHist(gray)

      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
      cl1 = clahe.apply(gray)

      # tmp = np.hstack((gray, equ, cl1))

      # white = Create_Mask(hsv, "white")

      # lower = np.array([30,30,60])
      # upper = np.array([90,255,255])

      # green = cv2.inRange(hsv, lower, upper)

      # _,new = cv2.threshold(gray, 120,255, cv2.THRESH_BINARY)
      # new = cv2.medianBlur(new, 3)

      # new[np.where(green == 255)] = 0

      # blur = image
      # blur = cv2.bilateralFilter(image, 3,50,50)
      # blur = cv2.medianBlur(blur, 5)
      # gray,_ = Cvt_All(blur)

      f2 = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 15, -1)
      # mask = Filter_Image(image)

      # f3 = cv2.medianBlur(f2, 3)
      # Magic(image, f2)


      cv2.imshow("image",image)
      # cv2.imshow("gray",gray)
      cv2.imshow("f2",f2)
      # cv2.imshow("hsv",hsv)
      # cv2.imshow("white",white)
      cv2.waitKey(0)