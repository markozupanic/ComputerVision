import cv2

#Canny detektor

image=cv2.imread("kamera2.jpg")
canny_image=cv2.Canny(image,100,200)
cv2.imshow("Original slika",image)
cv2.imshow("Canny slika",canny_image)
cv2.waitKey()
cv2.destroyAllWindows

























