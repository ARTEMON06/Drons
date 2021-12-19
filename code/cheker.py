import cv2
import numpy
from tensorflow.python.keras.models import load_model


cam = cv2.VideoCapture(0)

SIGN_CLASSIFICATION = ("Traffic is prohibited", "Pedestrian crossing", "Stop", "Emergency stop", "Straight", "Right", "Left")
MODEL = load_model("22_1.h5")
# MINB = 109
# MING = 68
# MINR = 139
MINB = 96
MING = 65
MINR = 183
MINSIZE_W = 65
MINSIZE_H = 65


def prediction(frame):
    image = numpy.expand_dims(frame, axis=0)
    image = numpy.array(image)
    pred = MODEL.predict(image)
    kk = numpy.argmax(pred, axis=1)
    if pred[0][kk[0]] > 0.90:

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (x, y-20)
        fontScale = 1
        fontColor = (0, 255, 0)
        lineType = 1

        cv2.putText(image_copy1, SIGN_CLASSIFICATION[kk[0]],
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        print(f"{SIGN_CLASSIFICATION[kk[0]]} - {pred[0][kk[0]]}")
    else:
        print("None")


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

while True:
    ret, frame = cam.read()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv = cv2.blur(hsv, (5, 5))

    mask = cv2.inRange(hsv, (MINB, MING, MINR), (255, 255, 255))

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=5)

    cv2.imshow("mask", mask)

    contours1, hierarchy1 = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_copy1 = frame.copy()

    if contours1:
        c = max(contours1, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w > MINSIZE_W and h > MINSIZE_H:

            x, y, w, h = cv2.boundingRect(c)
            # print(f'{w} {h}')
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = numpy.int0(box)
            # print(box)
            # cv2.drawContours(image_copy1, [box], 0, (0, 0, 255), 2)
            cv2.rectangle(image_copy1, (x, y), (x + w, y + h), (0, 255, 0), 2)


            # x_w = x + w
            # y_h = y + h
            # image_znak = frame[y:y_h, x:x_w]
            # cv2.imshow('znak', image_znak)

            peri = cv2.arcLength(box, True)
            approx = cv2.approxPolyDP(box, 0.02 * peri, True)

            if len(approx) == 4:
                docCnt = approx
                paper = four_point_transform(frame, docCnt.reshape(4, 2))
                cv2.imshow('paper', paper)
                prediction(cv2.resize(paper, (48, 48)))

                # prediction(cv2.resize(image_znak, (48, 48)))

    cv2.imshow('Simple approximation', image_copy1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
