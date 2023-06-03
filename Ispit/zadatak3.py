import cv2
import numpy as np

img=cv2.imread('oblici.png')
img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
circles=cv2.HoughCircles(img_grey,cv2.HOUGH_GRADIENT,1,minDist=100,param1=200,param2=20)

circles=circles.squeeze()
circles=circles.astype(np.uint16)
for circle in circles:
    cv2.circle(img,(circle[0],circle[1]),circle[2],(0,255,0),thickness=3)
    
cv2.imshow("Circles",img)
cv2.waitKey()
cv2.destroyAllWindows()









