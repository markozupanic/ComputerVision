
"""
1. Nadograditi algoritam tako da radi na video zapisu
2. Napraviti pauziranje video zapisa na tipku "p" (ostatak algoritma mora se nastaviti izvršavati na pauziranom frameu)
3. Preraditi algoritam da pokazuje slike u više redova (npr. 2 slike u jednom redu) (cv2.vconcat())
4. Dodati trackbare za canny threshold-e i prikazati canny sliku (cv2.Canny())
5. Dodati feature koji mislite da bi bio koristan

"""


import cv2
import numpy as np

cv2.namedWindow("filter")

low = np.array([0, 0, 0])
high = np.array([255, 255, 255])

canny_h=255
canny_l=0


def first_low_trackbar(val):
    global low
    global high
    low[0] = val
    high[0] = max(high[0], low[0]+1)
    cv2.setTrackbarPos("High-0", "filter", high[0])

def first_high_trackbar(val):
    global low
    global high
    high[0] = val
    low[0] = min(high[0]-1, low[0])
    cv2.setTrackbarPos("Low-0", "filter", low[0])

def second_low_trackbar(val):
    global low
    global high
    low[1] = val
    high[1] = max(high[1], low[1]+1)
    cv2.setTrackbarPos("High-1", "filter", high[1])

def second_high_trackbar(val):
    global low
    global high
    high[1] = val
    low[1] = min(high[1]-1, low[1])
    cv2.setTrackbarPos("Low-1", "filter", low[1])

def third_low_trackbar(val):
    global low
    global high
    low[2] = val
    high[2] = max(high[2], low[2]+1)
    cv2.setTrackbarPos("High-2", "filter", high[2])

def third_high_trackbar(val):
    global low
    global high
    high[2] = val
    low[2] = min(high[2]-1, low[2])
    cv2.setTrackbarPos("Low-2", "filter", low[2])
    
def canny_lower(val):
    global canny_l
    canny_l = val
    
def canny_higher(val):
    global canny_h
    canny_h = val

    


cv2.createTrackbar("Low-0", "filter", 0, 255, first_low_trackbar)
cv2.createTrackbar("Low-1", "filter", 0, 255, second_low_trackbar)
cv2.createTrackbar("Low-2", "filter", 0, 255, third_low_trackbar)
cv2.createTrackbar("High-0", "filter", 255, 255, first_high_trackbar)
cv2.createTrackbar("High-1", "filter", 255, 255, second_high_trackbar)
cv2.createTrackbar("High-2", "filter", 255, 255, third_high_trackbar)
cv2.createTrackbar("Canny-Lower", "filter", 0, 255, canny_lower)
cv2.createTrackbar("Canny-Higher", "filter", 255, 255, canny_higher)




#read image and resize

cap=cv2.VideoCapture("video1.mp4")
#image = cv2.imread("red_object.jpg")
#image = cv2.resize(image, (300, 300))

while True:
    
    ret,frame =cap.read()
    frame=cv2.resize(frame,(200,200))
    if frame is None:
        break
    
    
    #RGB filter
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    filteredRGB = cv2.inRange(imageRGB, low, high)
    filteredRGB = cv2.cvtColor(filteredRGB, cv2.COLOR_GRAY2BGR)

    #HSV filter
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    filteredHSV = cv2.inRange(imageHSV, low, high)
    filteredHSV = cv2.cvtColor(filteredHSV, cv2.COLOR_GRAY2BGR)

    #HLS filter
    imageHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    filteredHLS = cv2.inRange(imageHLS, low, high)
    filteredHLS =  cv2.cvtColor(filteredHLS, cv2.COLOR_GRAY2BGR)
    
    #Canny filter
    imageCanny=cv2.Canny(frame,canny_l,canny_h)
    filteredCanny=cv2.cvtColor(imageCanny,cv2.COLOR_GRAY2BGR)
    
    
    def concat_vh(list_2d):# return final image
        return cv2.vconcat([cv2.hconcat(list_h) 
            for list_h in list_2d])


    showIMG = concat_vh([[frame, filteredRGB,filteredRGB],
                      [filteredHSV, filteredHLS,filteredCanny]])

    cv2.imshow("filtershow", showIMG)
    if cv2.waitKey(1) == ord("q"):
        print(f"LOW: {low} | HIGH: {high}")
        break
    
    if cv2.waitKey(1)==ord("p"):
        cv2.waitKey()

cv2.destroyAllWindows()