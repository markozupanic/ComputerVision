import cv2


def get_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"X: {x} | Y: {y}")


cv2.namedWindow("frame")
cv2.setMouseCallback("frame", get_coordinates)

cap = cv2.VideoCapture("video3.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("frame", frame)
    cv2.waitKey(0)
