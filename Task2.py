import cv2

from Tools import *


def Filter_Image(image):
   canny = cv2.Canny(image,50,200)

   cv2.imshow("canny",canny)

   lines = cv2.HoughLinesP(image=canny,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=50,maxLineGap=30)

   for line in lines:
      x1,y1,x2,y2 = line[0]
      cv2.line(image, (x1,y1),(x2,y2), (0,255,0),2)

def main(files):
   for f in files:
      image = cv2.imread(f)

      blur = cv2.bilateralFilter(image, 15,50,50)

      blur = cv2.medianBlur(blur, 5)

      _,f1 = cv2.threshold(blur, 150,255, cv2.THRESH_BINARY)

      # mask = Filter_Image(blur)

      cv2.imshow("image",image)
      cv2.imshow("f1",f1)
      # cv2.imshow("equ",equ)
      cv2.waitKey(0)