import cv2
import numpy as np



src =np.array( [[470,470],
                [270,570],
                [920,551],
                [835,469]
    ], dtype = np.float32)


h = 400
w = 720
dst = np.array([
    [0, 0],
    [0, h],
    [w, h],
    [w, 0]
], dtype = np.float32)


cv2.namedWindow("frame")

cap = cv2.VideoCapture("video3.mp4")

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    warped_img = cv2.warpPerspective(frame, M, (w,h))
    unwarped = cv2.warpPerspective

    cv2.imshow("warped", warped_img)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)