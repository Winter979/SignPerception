import glob
import cv2

import numpy as np

from Settings import Settings as s


if __name__ == '__main__':
   files = glob.glob("{}/*".format(s.T1_Images))

   mser = cv2.MSER_create()

   for f in files:
      image = cv2.imread(f);

      pts = np.array(eval(args["coords"]), dtype = "float32")
      warped = four_point_transform(image, pts)

      cv2.imshow('image', image)
      cv2.imshow('warped', warped)
      cv2.waitKey(0)
      cv2.destroyAllWindows()



    