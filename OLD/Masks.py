import cv2
import numpy as np

def Black_Mask(hsv):
   

   lower_gray = np.array([0,0,0])
   upper_gray = np.array([255,180,120])
   mask = cv2.inRange(hsv, lower_gray, upper_gray)

   return ~mask

def Red_Mask(hsv):
   

   # RED
   lower_red = np.array([0,60,80])
   upper_red = np.array([30,255,255])
   mask_0 = cv2.inRange(hsv, lower_red, upper_red)

   lower_red = np.array([160,60,80])
   upper_red = np.array([180,255,255])
   mask_1 = cv2.inRange(hsv, lower_red, upper_red)   

   red_mask = mask_0 + mask_1

   return red_mask

def Green_Mask(hsv):
   
   lower_green = np.array([41, 60, 64])
   upper_green = np.array([90, 255, 255])
   green_mask = cv2.inRange(hsv, lower_green, upper_green)

   return green_mask

def Yellow_Mask(hsv):
   lower_yellow = np.array([21, 60, 64])
   upper_yellow = np.array([40, 255, 255])
   yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

   return yellow_mask

def White_Mask(hsv):
   lower_white = np.array([0, 0, 200])
   upper_white = np.array([180, 50, 255])
   white_mask = cv2.inRange(hsv, lower_white, upper_white)

   return white_mask

def Brown_Mask(hsv):
   lower_brown = np.array([5, 100, 20])
   upper_brown = np.array([30, 255, 200])
   brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

   return brown_mask