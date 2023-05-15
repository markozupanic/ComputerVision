#Metodom oduzimanja pozadine pokušajte izdvojiti crveni autić iz videa “red_car_moving.avi”.
#● Pojavljuju li se određene smetnje?
#● Primijenite odgovarajuća morfološka filtriranja kako bi uklonili šum

import cv2 as cv
import numpy as np


background_subtractor = cv.createBackgroundSubtractorMOG2()

capture = cv.VideoCapture("red_car_moving.mp4")

if not capture.isOpened():
    print("ERROR: Cannot open the video")
    exit(1)

while True:
    ret, frame = capture.read()

    if frame is None:
        break

    foreground_mask = background_subtractor.apply(frame)


    cv.imshow("Foreground Mask", foreground_mask)


    cv.imshow("Frame", frame)

    if cv.waitKey(1) == ord("q") or cv.waitKey(1) == ord("Q"):
        break

capture.release()
cv.destroyAllWindows()