import cv2
import numpy as np

counter = 0
src_temp = []

def get_coordinates(event, x, y, flags, param):
    global counter, src_temp, src
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x}, {y}],")
        #print(counter)
        if counter == 3:
            counter = 0
            src_temp.append([x,y])
            src = np.array(src_temp, dtype = np.float32)
            src_temp = []
            
            M = cv2.getPerspectiveTransform(src, dst)
            M_inv = cv2.getPerspectiveTransform(dst, src)
    
            warped_img = cv2.warpPerspective(frame, M, (w,h))
            unwarped = cv2.warpPerspective

            cv2.imshow("warped", warped_img)
            
        elif counter < 3:
            counter += 1
            src_temp.append([x,y])
        
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", get_coordinates)

cap = cv2.VideoCapture("video1.mp4")


src =np.array( [
[435, 503],
[150, 617],
[1269, 633],
[889, 503]

], dtype = np.float32)


h = 400
w = 720
dst = np.array([
    [0, 0],
    [0, h],
    [w, h],
    [w, 0]
], dtype = np.float32)



while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    cv2.imshow("frame", frame)
    cv2.waitKey(5)