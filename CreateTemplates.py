import cv2

def Arrows():
   image = cv2.imread("arrow.png")

   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   _,th = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

   res = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]


   x,y,w,h = cv2.boundingRect(cnts[0])      

   crop = th[y:y+h,x:x+w]
   crop = cv2.resize(crop,(50,50))

   drc = ['L','U','R','D']

   for d in drc:
      cv2.imwrite("templates/{}.jpg".format(d),crop)
      crop = cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE)

   cv2.imshow("crop",crop)
   cv2.waitKey(0)
   

def Numbers():
   image = cv2.imread("characters.png")

   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   _,th = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

   res = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]


   better = []

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)

      # Remove the non numbers
      if h < 50:
         continue

      better.append([x,c])

   better = sorted(better, key=lambda x: x[0])

   better = [ii[1] for ii in better]

   val = 0

   for c in better:

      x,y,w,h = cv2.boundingRect(c)

      crop = th[y:y+h,x:x+w]
      crop = cv2.resize(crop,(50,50))

      cv2.imwrite("templates/numbers/{}.jpg".format(val),crop)

      val += 1

def Alphabet():
   image = cv2.imread("alphabet2.png")

   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   _,th = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)

   res = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
   cnts,_ = res if len(res) == 2 else res[1:3]


   better = []

   for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      # print(h)
      # continue
      # Remove the non numbers
      if h > 21:
         better.append([x,c])

   better = sorted(better, key=lambda x: x[0])

   better = [ii[1] for ii in better]

   val = ord('A')

   for c in better:

      x,y,w,h = cv2.boundingRect(c)

      crop = th[y:y+h,x:x+w]
      crop = cv2.resize(crop,(50,50))

      cv2.imwrite("templates/lower/{}_l.jpg".format(chr(val)),crop)

      val += 1

if __name__ == "__main__":
   # Alphabet()
   Arrows()
   

      # cv2.imshow("crop",crop)
      # cv2.waitKey(0)
      # cv2.drawContours(image, [c], -1,(0,0,255),1)


   # cv2.imshow("image",image)
   # cv2.waitKey(0)