#8 Učitajte sliku “kamera1.png”, te detektirajte rubove pomoću Canny detektora rubova

import cv2

image=cv2.imread("kamera1.jpg")
canny_image=cv2.Canny(image,100,200)
cv2.imshow("Canny slika",canny_image)
cv2.imshow("Normalna slika",image)
cv2.waitKey()
cv2.destroyAllWindows()










