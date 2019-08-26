import cv2

from Tools import *
from NewTools import *

if __name__ == "__main__":
   files = Get_Files(part=1, test=False)

   for f in files:
      image = cv2.imread(f)

      _,t1 = cv2.threshold(image, 150,255, cv2.THRESH_BINARY)

      cv2.imshow("t1", t1)
      cv2.waitKey(0)