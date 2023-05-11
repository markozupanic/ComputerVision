import cv2
import numpy as np

#1.ućitavanje
#image = cv2.imread("kamera1.jpg",cv2.IMREAD_COLOR)
#cv2.imshow("Slika", image)
#cv2.waitKey()
#cv2.destroyAllWindows()

#2 spremanje
#image = cv2.imread("kamera1.jpg",cv2.IMREAD_GRAYSCALE)
#cv2.imwrite("kamera_1_grayscale.jpg",image)

#3 promjena veličine
#image = cv2.imread("kamera1.jpg",cv2.IMREAD_COLOR)
#image_resized=cv2.resize(image, None,fx=2,fy=2)
#image_resized=cv2.resize(image, (100,200))
#resized_img=np.zeros((700,300,3),dtype=np.int8)
#height,width,_=resized_img.shape
#cv2.resize(image, (width,height),resized_img)
#print("Velićina originalne slike: {}".format(image.shape))
#print("Velićina originalne slike: {}".format(resized_img.shape))
#print("Velićina povečane slike: {}".format(image_resized.shape))
#cv2.imshow("Originalna slika",image)
#cv2.imshow("OPovečana slika",image_resized)
#cv2.imshow("OPovečana slika",resized_img)
#cv2.waitKey()
#cv2.destroyAllWindows()

#4 flipanje
#image = cv2.imread("kamera1.jpg",cv2.IMREAD_COLOR)
#image_fliped=cv2.flip(image,0)#po x osi rotiranje
#cv2.imshow("Original image",image)
#cv2.imshow("Fipped image",image_fliped)
#cv2.waitKey()
#cv2.destroyAllWindows()

#5 dio slike u jednu boju 
#image = cv2.imread("kamera1.jpg",cv2.IMREAD_COLOR)
#image[0:70,0:100]=[0,0,255]
#cv2.imshow("Original shape",image)
#cv2.waitKey()
#cv2.destroyAllWindows()

#6samo B 
#image = cv2.imread("kamera1.jpg",cv2.IMREAD_COLOR)
#image[:,:,1:]=0
#cv2.imshow("Original shape",image)
#cv2.waitKey()
#cv2.destroyAllWindows()

#7 RGB u HSV
#image_gbr = cv2.imread("kamera1.jpg",cv2.IMREAD_COLOR)
#image_hvs=cv2.cvtColor(image_gbr,cv2.COLOR_BGR2HSV)
#cv2.imshow("Original shape",image_gbr)
#cv2.imshow("HSV shape",image_hvs)
#cv2.waitKey()
#cv2.destroyAllWindows()

#8 Bojanje u HSV-u i prebacivanje nazad u RGB
#image_gbr = cv2.imread("kamera1.jpg",cv2.IMREAD_COLOR)
#image_hvs=cv2.cvtColor(image_gbr,cv2.COLOR_BGR2HSV)
#image_hvs[:70,:100]=[0,255,255]
#imag_bgr=cv2.cvtColor(image_hvs,cv2.COLOR_HSV2BGR)
#cv2.imshow("Original shape",imag_bgr)
#cv2.waitKey()
#cv2.destroyAllWindows()

#9 BIinarizacija slike
#image_greyscale = cv2.imread("chair.png",cv2.IMREAD_GRAYSCALE)
#_,binary_image=cv2.threshold(image_greyscale, 230, 255, cv2.THRESH_BINARY) 
#cv2.imshow("Original shape",image_greyscale)
#cv2.imshow("Binary image",binary_image)
#cv2.waitKey()
#cv2.destroyAllWindows()


#10  filtriranje samo određene boje
















