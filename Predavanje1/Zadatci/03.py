#3.Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
#“kamera2.jpg”. Zaokrenite sliku po x i y osi, prikažite ju, te spremite pod nazivom
#“zaokrenuta_slika.jpg”.
import cv2 

image=cv2.imread("kamera2.jpg")
image_fliped=cv2.flip(image,-1)
cv2.imwrite("zaokrenuta_slika.jpg",image_fliped)
cv2.imshow("Flipana slika",image_fliped)
cv2.waitKey()
cv2.destroyAllWindows()






















