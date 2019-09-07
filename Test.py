import glob
import cv2

import numpy as np

from Settings import Settings as s


if __name__ == '__main__':
   files = glob.glob("{}/*".format(s.T1_Images))

   mser = cv2.MSER_create()

   for f in files:
      image = cv2.imread(f);

      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      
      regions,_ = mser.detectRegions(image)

      hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
      # cv2.polylines(vis, hulls, 1, (0, 255, 0))``

      mask = np.ones(image.shape[:2],dtype="uint8") * [255,255,255]

      cv2.drawContours(mask, hulls, -1, (255,0,0),-1)

      cv2.imshow('image', image)
      cv2.imshow('mask', mask)
      cv2.waitKey(0)
      cv2.destroyAllWindows()



    