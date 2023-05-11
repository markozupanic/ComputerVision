import cv2
import numpy as np

image=cv2.imread("kamera1.jpg")

image_blurred=cv2.blur(image,(5,5))
cv2.imshow("Originalna slika",image)
cv2.imshow("OBlurana slika",image_blurred)
cv2.waitKey()
cv2.destroyAllWindows()