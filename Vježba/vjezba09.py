#11- Bitwise Operations (bitwise AND, OR, NOT and XOR)

import cv2
import numpy as np 

img1=np.zeros((250,500,3),np.uint8)
img1=cv2.rectangle(img1,(200,0),(300,100),(255,255,255),-1)
img2=cv2.imread('half.jpg')
img2=cv2.resize(img2,(500,250))


#bitAnd=cv2.bitwise_and(img2,img1)
bitOr=cv2.bitwise_or(img2,img1)


cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('bitAnd',bitOr)
cv2.waitKey()
cv2.destroyAllWindows()

















