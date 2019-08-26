import cv2
import numpy as np
import imutils

from Tools import *



def Filter_Image(image):

   gray, hsv = Cvt_All(image)

   masks = []

   masks.append(Create_Mask(hsv, "white"))
   red_mask1 = Create_Mask(hsv, "red1")
   red_mask2 = Create_Mask(hsv, "red2")
   masks.append(red_mask1 + red_mask2)

   masks.append(Create_Mask(hsv, "brown"))
   masks.append(Create_Mask(hsv, "green"))
   masks.append(Create_Mask(hsv, "yellow"))
   masks.append(Create_Mask(hsv, "shadow"))
   masks.append(Create_Mask(hsv, "pink"))
   masks.append(Create_Mask(hsv, "p_red"))

   ultimate_mask = np.zeros(image.shape[:2], np.uint8)
   
   ii = 0
   for mask in masks:
      # cv2.imshow(str(ii), mask)
      ultimate_mask += mask
      ii += 1

   ultimate_mask = cv2.medianBlur(ultimate_mask,7)
   Trim_Edges(ultimate_mask)
   # cv2.imshow("ultimate", ultimate_mask)

   return ultimate_mask

def Extract_Contours(mask):

   cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)

   new = np.ones(mask.shape[:2],dtype="uint8") * 255

   image_area = mask.size

   b_cnts = []

   for c in cnts:
      area = cv2.contourArea(c)
      # print(area)
      if 0.02*image_area < area < 0.1*image_area:
         x,y,w,h = cv2.boundingRect(c)
         ratio = w/h
         if 1 < ratio < 2:
            cv2.drawContours(new, [c], -1, 0, -1)
            b_cnts.append(c)

   return new, b_cnts

def Get_Sign(full_image, full_mask, m_cnts):

   # masked = Apply_Mask_Image(image, mask)

   for m_c in m_cnts:
      x,y,w,h = cv2.boundingRect(m_c)

      mask = full_mask[y:y+h,x:x+w]
      image = full_image[y:y+h,x:x+w]
           
      temp = Apply_Mask_Image(image, mask)
      temp,_ = Cvt_All(temp)
      _, temp = cv2.threshold(temp, 150, 255, cv2.THRESH_BINARY)
      Trim_Edges(temp, color = 0, width=5)

      temp[np.where(mask != 0)] = 0

      Get_Letters(temp, image)

   # masked = Apply_Mask_Image(image, mask)
   # cv2.imshow("masked",masked)

def Get_Letters(full_mask, full_image):
   cnts = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)

   ii = 0

   for c in cnts:
      area = cv2.contourArea(c)
      # You found a letter (Probably)
      if area > 100:
         x,y,w,h = cv2.boundingRect(c)
         mask = full_mask[y:y+h,x:x+w]
         image = full_image[y:y+h,x:x+w]
         Get_Letter(mask, image, ii)
         ii += 1
         

def Get_Letter(mask, image, ii):
   cv2.imshow("letter-{}".format(ii), mask)

   cuts = 5

   h,w,_ = image.shape

   cell_width = w//cuts
   cell_height = h//cuts

   quads = []

   for ii in range(cuts):
      for jj in range(cuts):
         x = cell_width * ii
         y = cell_height * jj

         crop = mask[y:y+cell_height,x:x+cell_width]

         ratio = Get_Ratio(crop)

         quads.append(ratio)

   print(",".join("%.3f" % q for q in quads))

def Get_Ratio(image):

   cells = 0
   count = 0

   h,w = image.shape

   for x in range(w):
      for y in range(h):
         count += 1
         if image[y,x] == 255:
            cells += 1
      
   ratio = cells / count

   return ratio

def main(files):
   for f in files:
      image = cv2.imread(f)

      # blur = image.copy()
      blur = cv2.bilateralFilter(image, 15,50,50)

      dark_mask = ~Create_Mask(blur, "dark")

      mask = Filter_Image(blur)

      # Create the combined mask
      combined = dark_mask + mask
      # combined = cv2.medianBlur(combined, 7)
      combined[np.where(combined != 0)] = 255

      new_mask,cnts = Extract_Contours(combined)

      Get_Sign(image, new_mask, cnts)

      cv2.imshow("image",image)
      
      cv2.waitKey(0)
      cv2.destroyAllWindows()
