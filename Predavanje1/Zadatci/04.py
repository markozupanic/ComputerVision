#4 Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
#“kamera2.jpg”. Promijenite vrijednost piksela koji čine dio registracijske oznake automobila koji
#se nalazi ispred nas u žutu boju
import cv2
image=cv2.imread("kamera2.jpg")
image[326:329,215:228]=[66,250,220]
cv2.imshow("Obojana slika",image)
cv2.waitKey()
cv2.destroyAllWindows()











