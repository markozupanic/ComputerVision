import cv2
import numpy as np

img=cv2.imread("sum.png")
kernel=np.ones((5,5),dtype=np.uint8)
img_without_noise=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

cv2.imshow("Sa šumom",img)
cv2.imshow("Bez šuma",img_without_noise)
cv2.waitKey()
cv2.destroyAllWindows()


















