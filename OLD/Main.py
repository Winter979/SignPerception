import cv2


from Tools import *



if __name__ == '__main__':
   files = Get_Files()

   for f in files:
      image = cv2.imread(f)

      blur = cv2.GaussianBlur(image, (3,3), 0)

      # blur = cv2.medianBlur(image, 3)

      gray, hsv = Cvt_All(blur)

      mask = Filter_Dark(hsv, color="hsv")

      cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

      cnts = imutils.grab_contours(cnts)

      better_cnts = []

      for c in cnts:
         rect = cv2.minAreaRect(c)
         width = rect[1][0]
         height = rect[1][1]
            
         area = width * height

         # Estimate of area size
         if 10000 < area:
            ratio = width/height
            if 0.6 < ratio < 2:
               better_cnts.append(c)


      for c in better_cnts:
         x,y,w,h = cv2.boundingRect(c)
         
         item = image[y:y+h,x:x+w]

         # Clean_Crop(item)

         Refine_It(item)


      # t1 = image.copy()
      # cv2.drawContours(image, better_cnts, -1, (0,255,0), 2)

      cv2.imshow("image", image)
      cv2.imshow("mask", mask)
      # cv2.imshow("gray", gray)
      # cv2.imshow("edges", edges)
      # cv2.imshow("hsv", hsv)
      # Filter_Dark(hsv)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

   
