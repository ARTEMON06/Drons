import cv2
import numpy

# 17 65 82
MINB = 20
MING = 60
MINR = 100
MINSIZE_W = 40
MINSIZE_H = 55

cam = cv2.VideoCapture(0)


def order_points(pts):
    rect = numpy.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[numpy.argmin(s)]
    rect[2] = pts[numpy.argmax(s)]
    diff = numpy.diff(pts, axis=1)
    rect[1] = pts[numpy.argmin(diff)]
    rect[3] = pts[numpy.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = numpy.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def spot_light(c):
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = numpy.int0(box)
    peri = cv2.arcLength(box, True)
    approx = cv2.approxPolyDP(box, 0.02 * peri, True)

    if len(approx) == 4:
        paper = four_point_transform(frame, approx.reshape(4, 2))
        paper = cv2.resize(paper, (48, 48))

        counter_yellow = 0
        counter_red = 0
        counter_green = 0

        for i in range(10, 30):
            for j in range(10, 30):
                pixel = paper[i][j]
                print(pixel)
                if pixel[0] > 100 and pixel[1] > 200 and pixel[2] > 200:
                    counter_yellow += 1

                elif pixel[0] > 100 and pixel[1] > 100 and pixel[2] > 100:
                    counter_green += 1

                elif pixel[0] > 100 and pixel[1] > 90 and pixel[2] > 200:
                    counter_red += 1

        cv2.rectangle(image_copy1, (x, y), (x + w, y + h), (255, 255, 255), 3)

        if counter_green > 350 or counter_red > 350 or counter_yellow > 350:
            if counter_green > 350:
                cv2.rectangle(image_copy1, (x, y), (x + w, y + h), (0, 255, 0), 7)
                # print('green')
            if counter_red > 350:
                cv2.rectangle(image_copy1, (x, y), (x + w, y + h), (0, 0, 255), 7)
                # print('red')
            if counter_yellow > 350:
                cv2.rectangle(image_copy1, (x, y), (x + w, y + h), (0, 255, 255), 7)
                # print('yellow')
        else:
            pass
            # print('nothing')

        # print(f"counter_red, counter_yellow, counter_green")


while True:
    ret, frame = cam.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv = cv2.blur(hsv, (5, 5))

    mask = cv2.inRange(hsv, (MINB, MING, MINR), (255, 255, 255))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)

    contours1, hierarchy1 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours1:
        image_copy1 = frame.copy()

        c = max(contours1, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > MINSIZE_W and h > MINSIZE_H:
            spot_light(c)

        cv2.imshow('Simple approximation', image_copy1)
    cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
