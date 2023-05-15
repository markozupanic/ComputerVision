#3 Pomoću Houghove transformacije pokušajte detektirati prometni znak na slici “kamera4.png”.
import cv2
import numpy as np
img=cv2.imread("kamera4.png")
img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_stop=cv2.HoughCircles(img_grey,cv2.HOUGH_GRADIENT,1,minDist=750,param1=90,param2=120)

img_stop=img_stop.squeeze()
img_stop=img_stop.astype(np.uint16)
for circle in img_stop:
    cv2.circle(img,(circle[0],circle[1]),circle[2],(0,255,0),thickness=3)
    
cv2.imshow("Slika siva",img)
cv2.waitKey()
cv2.destroyAllWindows()
