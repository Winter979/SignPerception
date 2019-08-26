import cv2

from Tools import *
from NewTools import *

def Find_Blobs(image, mask):

   here = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

   detector = cv2.SimpleBlobDetector()
   keypoints = detector.detect(here)
 
   im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
   
   cv2.imshow("Keypoints", im_with_keypoints)

def Find_Contours(image, masked_image, mask):

   canny = cv2.Canny(masked_image,250,255)

   # gray, hsv = Cvt_All(masked_image)

   cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   cnts = imutils.grab_contours(cnts)

   better_cnts = []

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
            
      area = w*h

      # Estimate of area size
      if 8000 < area < 30000:
         ratio = w/h   
         if 1 < ratio < 2:
            better_cnts.append(c)


   ii = 0
   for c in better_cnts:
      x,y,w,h = cv2.boundingRect(c)
      
      item = image[y:y+h,x:x+w]
      cv2.imshow("item{}".format(ii), item)

      ii += 1

   cv2.drawContours(masked_image, better_cnts, -1, (0,255,0), 5)

def Clean_Up(masked_image, mask):
   canny = cv2.Canny(mask,250,255)

   # gray, hsv = Cvt_All(masked_image)

   cnts = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
   cnts = imutils.grab_contours(cnts)

   better_cnts = []

   iH, iW,_ = masked_image.shape

   new_mask = np.ones((iH, iW), dtype="uint8") * 255

   print("="*20)
   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
            
      area = w*h

      # Estimate of area size
      if area < 8000:
         cv2.drawContours(new_mask, [c], -1,0,-1)
      elif iW * 0.5 < w  or w < iW * 0.1:
         cv2.drawContours(new_mask, [c], -1,0,-1)
      elif iH * 0.3 < h or h < iH * 0.1:
         cv2.drawContours(new_mask, [c], -1,0,-1)
      else:
         ratio = w/h
         if ratio > 10 or ratio < 0.5:
            cv2.drawContours(new_mask, [c], -1,0,-1)

   new_mask = mask | ~new_mask

   cv2.imshow("new_mask",new_mask)

if __name__ == "__main__":
   files = Get_Files(part=1, test=False)

   for f in files:
      image = cv2.imread(f)

      mask = Better_Mask(image)
      
      masked_image = image.copy()
      masked_image[mask != 0] = [255,255,255]

      Clean_Up(masked_image, mask)

      # Find_Contours(image, masked_image, mask)

      

      # cv2.imshow("new", new)
      # cv2.imshow("new2", new2)
      # cv2.imshow("canny", canny)
      # cv2.imshow("dark", dark)
      cv2.imshow("mask",mask)
      cv2.imshow("masked_image",masked_image)
      cv2.imshow("image",image)
      
      cv2.waitKey(0)
      cv2.destroyAllWindows()
