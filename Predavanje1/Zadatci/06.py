#Pomoću OpenCV-a učitajte sliku “chair.png”. Izvršite binarizaciju slike tako da bijelim pikselima
#budu označeni samo pikseli koji čine stolicu.
import cv2


image=cv2.imread("chair.png",cv2.IMREAD_GRAYSCALE)
_,binary_image=cv2.threshold(image,230,255,cv2.THRESH_BINARY_INV)
cv2.imshow("Original shape",image)
cv2.imshow("Binary image",binary_image)
cv2.waitKey()
cv2.destroyAllWindows()





















