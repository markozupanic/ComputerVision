import cv2
import numpy as np

image=cv2.imread("autocesta.jpg")
#image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

image_canny=cv2.Canny(image,100,200)
lines=cv2.HoughLinesP(image_canny, 2, np.pi/180,100,minLineLength=100,maxLineGap=10)
lines=lines.squeeze()
for line in lines:
    cv2.line(image,(line[0],line[1]),(line[2],line[3]),(0,255,0),thickness=3)

cv2.imshow("Canny picture",image_canny)
cv2.imshow("Detected lines",image)
cv2.waitKey()
cv2.destroyAllWindows()






















