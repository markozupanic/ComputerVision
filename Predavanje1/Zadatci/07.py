#Učitajte sliku “stop_sign.jpg”. Metodom filtriranja boje detektirajte znak stop. Rješenje zadatka
#prikazano je na slijedećoj slici.

import cv2
import numpy as np

image_bgr=cv2.imread("stop_sign.jpg")
image_hsv=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2HSV)

lower_red_boundry=np.array([0,120,50])
upper_red_boundry=np.array([1,200,255])
red_mask=cv2.inRange(image_hsv,lower_red_boundry,upper_red_boundry)
rec=cv2.boundingRect(red_mask)

cv2.rectangle(red_mask,rec,color=100,thickness=2)
red_mmask_rec=cv2.bitwise_and(image_bgr,image_bgr,mask=red_mask)

cv2.imshow("Original with rectangle",red_mmask_rec)
cv2.imshow("Original picture",image_bgr)
cv2.imshow("Red mask",red_mask)
cv2.waitKey()
cv2.destroyAllWindows()












