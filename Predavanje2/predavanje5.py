import cv2
import numpy as np

backSub=cv2.createBackgroundSubtractorMOG2()
capture=cv2.VideoCapture("red_car_moving.mp4")

if not capture.isOpened():
    print("Error:cannot open video")
    exit(1)

print(capture.get(cv2.CAP_PROP_FPS))

while True:
    ret,frame =capture.read()
    if frame is None:
        break
    cv2.imshow("Frame",frame)
    if cv2.waitKey(1)==ord('q'):
        break
    mask=backSub.apply(frame)
    
cv2.imshow("Mask",mask)
capture.release()   
cv2.destroyAllWindows()


































