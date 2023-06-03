import cv2
import numpy as np

img = cv2.imread('street-signs-1.jpg')
img = cv2.resize(img, (1280, 720))

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

l_red_boundry = np.array([0, 120, 50])
u_red_boundry = np.array([1, 200, 255])

red_mask = cv2.inRange(img_hsv, l_red_boundry, u_red_boundry)
rec = cv2.boundingRect(red_mask)
cv2.rectangle(img, rec, color=(0, 255, 0), thickness=2)
red_mask_rec = cv2.bitwise_and(img, img, mask=red_mask)

font = cv2.FONT_HERSHEY_SIMPLEX
text = 'Znak stop'
cv2.putText(red_mask_rec, text, (100, 150), font, 1, (0, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('Slika sa znakom stop i tekstom', red_mask_rec)
cv2.waitKey(0)
cv2.destroyAllWindows()
