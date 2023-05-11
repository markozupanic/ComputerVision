#2.Pomoću OpenCV-a učitajte sliku koja je snimljena s kamere automobila pod nazivom
#“kamera2.jpg”. Ispišite na ekran dimenzije slike. Promijenite veličinu slike na 1000x650.
#Prikažite sliku s promijenjenom veličinom, te ispišite na ekran novu veličinu slike.
import cv2

image=cv2.imread("kamera2.jpg")

print(f"Početna slika: {image.shape}")

image_resized=cv2.resize(image,(1000,650))
cv2.imshow("Slika original",image)
cv2.imshow("Slika",image_resized)
print(f"Uvečana slika:{image_resized.shape}")
cv2.waitKey()
cv2.destroyAllWindows








