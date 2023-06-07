'''
Zadatak 1 - implementacija osnovnog LDWS 

'''

import numpy as np
import cv2
import math


# TODO: Napisite funkciju za filtriranje po boji u HLS prostoru
# ulaz je slika u boji, funkcija vraca binarnu sliku te maske za bijelu, zutu boju i ukupnu masku
def filterByColor(frame):
    # TODO: Pretvorite sliku iz BGR u HLS
    frameHLS = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # TODO: Definirajte granice za bijelu boju te kreirajte masku pomocu funkcije cv2.inRange
    white_lower = np.array([0, 200, 0])
    white_upper = np.array([180, 255, 255])
    white_mask = cv2.inRange(frameHLS, white_lower, white_upper)

    # TODO: Definirajte granice za zutu boju te kreirajte masku pomocu funkcije cv2.inRange
    yellow_lower = np.array([20, 0, 100])
    yellow_upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(frameHLS, yellow_lower, yellow_upper)

    # TODO: Kombinirajte obje maske pomocu odgovarajuce logicke operacije (bitwise)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # TODO: Filtirajte sliku pomocu dobivene maske koristei odgovarajucu logicku operaciju (bitwise)
    filteredROI = cv2.bitwise_and(frame, frame, mask=mask)

    return white_mask, yellow_mask, mask, filteredROI


# TODO: Napisite funkciju za detekciju rubova; funkcija vraca binarnu sliku s detektiranim rubovima
def detectEdges(frame):
    image = cv2.Canny(frame, 100, 250)
    return image


# TODO: Napisite funkciju za pronalazenje pravaca lijeve i desne kolnice oznake
# ulaz je binarna slika, a izlaz dvije liste koje sadrze pravce koji pripadaju lijevoj odnosnoj desnoj kolnickoj oznaci
def findLines(canny, frameROI):
    # TODO: Koristite cv2.HoughLinesP() kako biste dobili linije na slici
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, 15, minLineLength=10, maxLineGap=200)

    # TODO: Pronadite od svih linija one koje predstavljaju lijevu odnosno desnu uzduznu kolnicku oznaku
    linesLeft = []
    linesRight = []
    
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            #TODO: pretvorite točke (x1, y1) i (x2, y2) u funkciju pravca y=ax+b
            #pomoc: https://keisan.casio.com/exec/system/1223508685
            if y1 == y2:
                y1+=1
            if x1 == x2:
                x1+=1

            a = (y2-y1)/(x2-x1)
            b = y1 - a*x1
            x_val = (canny.shape[0] -b)/a
            line_angle = math.atan2((y2-y1), (x2-x1)) * 180/np.pi

            #TODO: definirajte granice za prikaz linija vozne trake
            if x_val > 150 and x_val < 1200:
                if line_angle > 10 and line_angle<90:
                    #desna linija
                    if x_val > (canny.shape[1]/2)-200 and x_val <(canny.shape[1]/2)+200:
                        cv2.line(frameROI, (x1, y1), (x2, y2), (255,0,255), 2)
                        linesRight.append([a, b, 1, x_val, line_angle])

                    else:
                        cv2.line(frameROI, (x1, y1), (x2, y2), (255,0,0), 2)
                        linesRight.append([a, b, 0, x_val, line_angle])

                if line_angle < -10 and line_angle > -90:
                    #lijeva linija
                    if x_val > (canny.shape[1]/2)-200 and x_val <(canny.shape[1]/2)+200:
                        cv2.line(frameROI, (x1, y1), (x2, y2), (255,0,255), 2)
                        linesLeft.append([a, b, 1, x_val, line_angle])

                    else:
                        cv2.line(frameROI, (x1, y1), (x2, y2), (255,0,0), 2)
                        linesLeft.append([a, b, 0, x_val, line_angle])

    except:
        linesRight = []
        linesLeft = []

    return linesRight, linesLeft

# TODO: Napisite funkciju koja oznacava sa zelenom povrsinom voznu traku (podrucje unutar pravaca) te ispisuje upozorenje na originalni ulazni frame
def drawLane_math(linesLeft, linesRight, frameToDraw):
    if (len(linesLeft) < 1 or len(linesRight) < 1) or ((linesLeft[0][2] == 1) or (linesRight[0][2] == 1)): 
        putInfoImg(frameToDraw, "PRIJELAZ LINIJE!", (300, 300))
    else:
        # [a, b, 1, x_val, line_angle]
        # yMin = (height//2)+80
        # yMax = height-100

        angle_left = math.radians(180 - linesLeft[0][4])
        angle_right = math.radians(180 - linesRight[0][4])
        y_top = frameToDraw.shape[0] // 2 + 80
        y_bottom = frameToDraw.shape[0] - 100

        hypot_left = abs(y_top - y_bottom) / math.sin(angle_left)
        hypot_right = abs(y_top - y_bottom) / math.sin(angle_right)

        x1 = linesLeft[0][3] + hypot_left * math.cos(angle_left)
        x2 = linesRight[0][3] + hypot_right * math.cos(angle_right)
        
        pt1 = (x1, y_top)
        pt2 = (x2, y_top)
        pt3 = (linesRight[0][3], y_bottom)
        pt4 = (linesLeft[0][3], y_bottom) 
        points = np.int32([pt1, pt2, pt3, pt4])

        cv2.fillPoly(frameToDraw, [points], (0,255,0))


def drawLane(linesLeft, linesRight, frameToDraw):

    ymin = 0
    ymax = frameToDraw.shape[0]

    if linesLeft and linesRight:

        x1_1 = int((ymin - linesLeft[0][1]) / linesLeft[0][0])
        x1_2 = int((ymax - linesLeft[0][1]) / linesLeft[0][0])
            
        x2_1 = int((ymin - linesRight[0][1]) / linesRight[0][0])
        x2_2 = int((ymax - linesRight[0][1]) / linesRight[0][0])

        if linesLeft[0][2] == 0 and linesRight[0][2] == 0:
            contours = np.array([[x1_1,ymin+yMin], [x2_1,ymin+yMin], [x2_2, ymax+yMax], [x1_2,ymax+yMax]])
            overlay = frameToDraw.copy()

            cv2.fillPoly(overlay, [contours], color=(0, 255, 100))
            cv2.addWeighted(overlay, 0.35, frameToDraw, 1 - 0.35, 0, frameToDraw)

    if linesLeft:
        if linesLeft[0][2] == 1:
            putInfoImg(frameToDraw, "Upozorenje!", (int(width/2)-140, 150), (0, 0, 255))
    
    if linesRight:
        if linesRight[0][2] == 1:
            putInfoImg(frameToDraw, "Upozorenje!", (int(width/2)-140, 150), (0, 0, 255))
        
    return frameToDraw

def putInfoImg(img, text, loc, color=(0, 255, 255)):
    cv2.putText(img, 
                text, 
                (loc[0], loc[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                color, 
                2, 
                cv2.LINE_4)


pathVideos = 'videos/'
videoName  = 'video2.mp4'


# TODO: Otvorite video pomocu cv2.VideoCapture
cap = cv2.VideoCapture("video1.mp4")

# TODO: Spremite sirinu i visinu video okvira u varijable width i height
_, frame = cap.read()
height, width = frame.shape[:2]

# TODO: Otvorite prozore za prikaz video signala i ostale rezultate (neka bude tipa cv2.WINDOW_NORMAL)

# ovdje definirajte sve ostale varijable po potrebi koje su vam potrebne za razvoj rjesenja
k = 0
yMin = (height//2)+80
yMax = height-100

while True:
    e1 = cv2.getTickCount()
    # TODO: Ucitaj video okvir pomocu metode read, povecaj k za jedan ako je uspjesno ucitan
    ret, frame = cap.read()
    if ret:
        k+=1
    else:
        break
    
    # TODO: Kreiraj regiju od interesa (RoI) izdvajanjem dijela numpy polja koje predstavlja 
    #       video okvir

    frameROI = frame[yMin:yMax, :, :]

    # TODO: Pozovite funkciju za filtriranje po boji RoI-a
    white_mask, yellow_mask, mask, filteredROI = filterByColor(frameROI)

    # TODO: Pozovite funkciju za detekciju rubova na filtriranoj slici kako bi ste smanjili
    #       kolicinu piksela koji se dalje procesiraju
    canny = detectEdges(filteredROI)

    # TODO: Pozovite funkciju za pronalazak pravaca lijeve i desne linije na slici s rubovima
    linesRight, linesLeft = findLines(canny, frameROI)

    # TODO: Pozovite funkciju za prikaz vozne trake na ulaznom video okviru
    drawLane(linesLeft, linesRight, frame)

    # TODO: Prikazi video okvir pomocu cv2.imshow(); i sve ostale medjurezultate kada ih napravite
    cv2.imshow("frame", frame)
    cv2.imshow("frame ROI", frameROI)
    cv2.imshow("mask", mask)
    cv2.imshow("filtered roi", filteredROI)
    cv2.imshow("canny", canny)

    key =  cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break
    
    # TODO: Ovdje ispisite vrijeme procesiranja jednog video okvira
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()


# TODO: Ovdje unistite sve prozore i oslobodite objekt koji je kreiran pomocu cv2.VideoCapture
cv2.destroyAllWindows()