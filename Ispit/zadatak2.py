import cv2 
import numpy as np


img=cv2.imread('2_semafor_zuto.png')
img=cv2.resize(img,(1280,720))

img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

l_yelow_boundry=np.array([28,200,90])
u_yelow_boundry=np.array([31,220,255])

yelow_mask=cv2.inRange(img_hsv,l_yelow_boundry,u_yelow_boundry)

rec=cv2.boundingRect(yelow_mask)

cv2.rectangle(img,rec,color=(0,255,0),thickness=2)

yelow_mask_rec=cv2.bitwise_and(img,img,mask=yelow_mask)



cv2.imshow("zuta maska",yelow_mask)
cv2.imshow("slika",img)
cv2.waitKey()
cv2.destroyAllWindows()



