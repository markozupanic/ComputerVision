#1 omoću Houghove transformacije detektirajte i nacrtajte pravce na slici “autocesta.jpeg”. Među
#detektiranim pravcima moraju biti detektirane i linije voznih traka.
#2 #Primijetite nedostatke prethodnog zadatka. Teško je detektirati samo linije vozne trake bez
#detekcije i ostalih linija na slici. Metodom maskiranja slike, pokušajte izdvojiti samo linije voznih
#traka. Preporučeno je da kao masku koristite gornju polovicu elipse u donjoj polovici slike.
import cv2
import numpy as np

image=cv2.imread("autocesta.jpg")
image_canny=cv2.Canny(image,100,200)
height,width=image_canny.shape


img=np.zeros((height,width),dtype=np.uint8)

cv2.ellipse(img,
            (width//2,height),
            axes=(450,300),
            angle=0,
            startAngle=130,
            endAngle=410,
            color=255,
            thickness=-1)


mask_image=cv2.bitwise_and(image_canny,img)
lines=cv2.HoughLinesP(mask_image, 2, np.pi/180,100,minLineLength=100,maxLineGap=10)
lines=lines.squeeze()
for line in lines:
    cv2.line(image,(line[0],line[1]),(line[2],line[3]),(0,255,0),thickness=3)



cv2.imshow("Nova slika",mask_image)
cv2.imshow("Canny picture",image_canny)
cv2.imshow("Detected lines",image)
cv2.waitKey()
cv2.destroyAllWindows()

















