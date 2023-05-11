import cv2 
import numpy as np 
slika_bgr = cv2.imread('kamera1.jpg') 
slikahsv = cv2.cvtColor(slika_bgr, cv2.COLOR_BGR2HSV) 
zelena_donja_granica = np.array([75, 80, 80]) 
zelena_gornja_granica = np.array([88, 255, 255]) 
maska = cv2.inRange(slikahsv, zelena_donja_granica, zelena_gornja_granica)
filtrirana_slika_hsv = cv2.bitwise_and(slikahsv, slikahsv,mask=maska) 
filtrirana_slika_bgr = cv2.cvtColor(filtrirana_slika_hsv,cv2.COLOR_HSV2BGR) 
cv2.imshow("Originalna slika", slika_bgr) 
cv2.imshow("Maska", maska) 
cv2.imshow("Filtrirana slika", filtrirana_slika_bgr) 
cv2.waitKey() 
cv2.destroyAllWindows() 
