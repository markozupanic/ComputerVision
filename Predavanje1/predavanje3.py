import cv2
import numpy as np

binary_image=np.zeros((200,200),dtype=np.uint8)
binary_image[40:160,30:110]=255
rec= cv2.boundingRect(binary_image)
cv2.rectangle(binary_image,rec,color=100,thickness=5)#na bgr slici color=(100,100,100)

cv2.imshow("Originalna slika",binary_image)
cv2.waitKey()
cv2.destroyAllWindows()




























