import cv2 
import numpy as np 

img=cv2.imread("kamera4.png")

img_grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

mask=np.zeros_like(img_grey)
height,width=img_grey.shape

x1=width//2
y1=0
x2=width-1
y2=0
x3=width-1
y3=height-1
x4=width//2
y4=height-1

maska_points=np.array([[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]])

mask=cv2.fillPoly(mask,maska_points,255)
mask_image=cv2.bitwise_and(img_grey,mask)
cv2.imshow("Masked image",mask_image)
cv2.waitKey()
cv2.destroyAllWindows()



