import cv2
import numpy as np

def main(files):
   for f in files:
      image = cv2.imread(f)

