#5 Ponovite prethodni zadatak, ali boju izmijenite koristeći HSV format.
import cv2

image_gbr=cv2.imread("kamera2.jpg",cv2.IMREAD_COLOR)
image_hsv=cv2.cvtColor(image_gbr,cv2.COLOR_BGR2HSV)
image_hsv[326:329,215:228]=[35,255,255]
image_back_to_gbr=cv2.cvtColor(image_hsv,cv2.COLOR_HSV2BGR)
cv2.imshow("Normalna slika",image_back_to_gbr)
#cv2.imshow("Obojana hsv slika",image_hsv)
cv2.waitKey()
cv2.destroyAllWindows()