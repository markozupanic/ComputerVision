'''
LDWS koji koristi transformaciju perspektive
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt


def plotArea(image, pts):
    for i in range(0,4):
        cv2.circle(image, (pts[i,0],pts[i,1]), radius=5, color=(255, 0, 0), thickness=-1)
    
    cv2.line(image, (pts[0,0], pts[0,1]), (pts[1,0],pts[1,1]), (0, 255, 0), thickness=3)
    cv2.line(image, (pts[1,0], pts[1,1]), (pts[2,0],pts[2,1]), (0, 255, 0), thickness=3)
    cv2.line(image, (pts[2,0], pts[2,1]), (pts[3,0],pts[3,1]), (0, 255, 0), thickness=3)
    cv2.line(image, (pts[3,0], pts[3,1]), (pts[0,0],pts[0,1]), (0, 255, 0), thickness=3)

    
def putInfoImg(img, text, loc):
    cv2.putText(img, 
                text, 
                (loc[0], loc[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)


# TODO: napisite funkciju za filtriranje po boji u HLS prostoru
# ulaz je slika u boji, funkcija vraca binarnu sliku te maske za bijelu, zutu boju i ukupnu masku
# metode: cv2.cvtColor, cv2.inRange, cv2.bitwise_or, cv2.bitwise_and
def filterByColor(image):
    imageHLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    white_lower = np.array([0, 200, 0])
    white_upper = np.array([180, 255, 255])
    white_mask = cv2.inRange(imageHLS, white_lower, white_upper)

    yellow_lower = np.array([20, 0, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(imageHLS, yellow_lower, yellow_upper)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    filteredImage = cv2.bitwise_and(image, image, mask=mask)

    return filteredImage, white_mask, yellow_mask, mask


# TODO: Napisite funkciju koja detektira dva maksimuma u sumi binarne slike po "vertikali",
# numpy metode: sum i argmax
# koristite samo donju polovicu binarne slike kako biste tocnije detektirali pocetak krivulje
def getTwoPeaks(binary_img):
    bottom = binary_img[height//2:, :]
    suma = np.sum(bottom, axis=0)

    x1 = np.argmax(suma)

    x1_1 = max(0, x1-150)
    x1_2 = min(x1+150, len(suma)-1)

    suma[x1_1:x1_2] = 0

    x2 = np.argmax(suma)

    if x1<x2:
        xl, xr = x1, x2
    else:
        xl, xr = x2, x1

    return xl, xr




def displayOriginal(original_img, left_x, right_x, M):
    lane = np.array([
    [left_x, 0],
    [left_x, 720],
    [right_x, 720],
    [right_x, 0]],
    dtype = np.float32)

    dst_lane = cv2.perspectiveTransform(np.array([lane]), M)
    plotArea(original_img, dst_lane[0].astype(np.int32))

def displayWarped(img, left_x, right_x):
    lane = np.array([
    [left_x, 0],
    [left_x, 720],
    [right_x, 720],
    [right_x, 0]],
    dtype = np.int32)

    plotArea(img, lane)


pathResults = 'results/'
pathVideos = 'videos/'
videoName  = 'video1.mp4'

#TODO: kreirati opencv prozore, cv2.namedWindow()


#TODO: otvoriti video i dohvatiti width i height
cap = cv2.VideoCapture(videoName)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


#TODO: pronaci 4 tocke za transformaciju perspektive
src = np.array([
    [300,583],
    [170,684],
    [1190,666],
    [1035,585]
], dtype=np.float32)

dst = np.array([
    [0, 0],
    [0, height],
    [width, height],
    [width, 0]
], dtype = np.float32)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

k = 1
time = 1

while(True):
    e1 = cv2.getTickCount()

    # TODO: Ucitaj frame i provjeri dali je uspjesno ucitan
    ret, frame = cap.read()

    if not ret:
        break
    
    # za debuggiranje 
    #plotArea(frame, src)

    # TODO: napravi transformaciju perspektive
    warped_image = cv2.warpPerspective(frame, M, (width, height))

    # TODO: Pozovite funkciju za filtriranje po boji nad ulaznim okvirom
    filteredImage, white_mask, yellow_mask, mask = filterByColor(warped_image)

    # TODO: Pozovite funkciju koja detektira dva maksimuma u sumi binarne slike po "vertikali" (histogramu)
    xl, xr = getTwoPeaks(mask)

    # TODO: Pozovite fuknkciju koja prikazuje detektiranu voznu traku na filtriranoj slici
    displayWarped(warped_image, xl, xr)

    # TODO: Pozovite funkciju koja prikazuje detektiranu voznu traku na originalnoj slici    
    displayOriginal(frame, xl, xr, M_inv)

    # TODO: Prikazite okvir pomocu cv2.imshow(); i sve ostale medjurezultate kada ih napravite
    cv2.imshow("frame", frame)
    cv2.imshow("warped", warped_image)
    cv2.imshow("mask", mask)

    # za debuggiranje

    # plt.figure(1)
    # x_array = range(0,mask.shape[1])
    # column_sums = mask[int(mask.shape[0]/2):,:].sum(axis=0)
    # plt.plot(x_array, column_sums)
    # plt.xlabel("x pozicija")
    # plt.ylabel("suma")
    # plt.show(block=False)
   

    # #prikaz informacija na videu
    putInfoImg(frame, "frame: " + str(k), ((50,50)))
    putInfoImg(frame, "FPS: " + str(int(1/time)), ((50,100)))

    #pauziranje i izlaz iz videa
    key =  cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    elif key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            if key2 == ord('p'):
                break
            if key2 == ord('q'):
                break
    
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    

cap.release()
cv2.destroyAllWindows()