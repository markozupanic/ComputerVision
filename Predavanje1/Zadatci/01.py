#1.Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
#“kamera2.jpg”. Sliku učitajte u nijansama sive boje. Ispišite na ekran dimenzije slike (visinu i
#širinu). Prikažite sliku, te spremite ju pod nazivom “kamera2_nijanse_sive.jpg”.
import cv2

image=cv2.imread("kamera2.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imwrite("kamera2_nijanse_sive.jpg",image)
cv2.imshow("Slika", image)
print(f"Visina: {image.shape[0]} Širina:{image.shape[1]}")
cv2.waitKey()
cv2.destroyAllWindows()


















