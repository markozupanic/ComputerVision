import cv2 
import numpy as np 

height=500
width=700
img=np.zeros((height,width),dtype=np.uint8)

cv2.ellipse(img,
            (width//2,height//2),
            axes=(width//2,height//2),
            angle=0,
            startAngle=180,
            endAngle=360,
            color=255,
            thickness=-1)
cv2.imshow("Elipse ",img)
cv2.waitKey()
cv2.destroyAllWindows()








