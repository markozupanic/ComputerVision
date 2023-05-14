import cv2
import numpy as np

#filtriranje samo određene boje(pronalazk semafora)

image_bgr=cv2.imread("kamera1.jpg")
image_hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)

lower_green_boundary=np.array([75,80,80])
upper_green_boundary=np.array([88,255,255])
green_mask=cv2.inRange(image_hsv,lower_green_boundary,upper_green_boundary)

#Dodavanje filtera na BGR sliku
green_mask_filtered=cv2.bitwise_and(image_bgr,image_bgr,mask=green_mask)
cv2.imshow("Green mask filter",green_mask_filtered)
#samo ovo dv

cv2.imshow("Original picture",image_bgr)
cv2.imshow("Green mask",green_mask)
cv2.waitKey()
cv2.destroyAllWindows()










