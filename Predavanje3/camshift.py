import cv2
import numpy as np

capture=cv2.VideoCapture('red_car_moving.mp4')
ret,frame=capture.read()

x,y,w,h=233,202,34,20

track_window=(x,y,w,h)

roi=frame[y:y+h,x:x+w]
roi_hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_mask=cv2.inRange(roi_hsv,np.array([170,50,50]),np.array([180,255,255]))

roi_hist=cv2.calcHist([roi_hsv],[0],roi_mask, [180],[0,180])
roi_hist=cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_criteria=(cv2.TermCriteria_EPS | cv2.TermCriteria_COUNT,10,1)

while True:
    ret,frame=capture.read()
    if ret :
        frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        back_projected_img=cv2.calcBackProject([frame_hsv],[0],roi_hist,[0,180],1)
        
        rotated_rect,track_window=cv2.CamShift(back_projected_img,track_window,term_criteria)
        points=cv2.boxPoints(rotated_rect)
        points=np.int32(points)
        cv2.polylines(frame,[points],True,(0,255,0),3)
        cv2.imshow("Detection",frame)
        if cv2.waitKey(2) ==ord('q'):
            break
     
    else:
        break

capture.release()
cv2.destroyAllWindows()