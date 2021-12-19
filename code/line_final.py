import numpy
from cv2 import cv2

LINE_WIDTH_NORMAL = (120, 170)
CAMERA_RESOLVE = (480, 640)

camera = cv2.VideoCapture(0)
camera_height = CAMERA_RESOLVE[0]
camera_width = CAMERA_RESOLVE[1]
camera_center = camera_width/2
last_line_center = camera_center


def draw_dot(frame, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (position - 35, camera_height - 33)
    fontScale = 0.7
    fontColor = (0, 0, 255)
    lineType = 1

    cv2.putText(frame, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    start_point = (position - 2, camera_height - 18)
    end_point = (position + 2, camera_height - 6)
    color = (0, 0, 255)
    thickness = -1
    cv2.rectangle(frame, start_point, end_point, color, thickness)


while True:
    ret, frame = camera.read()
    # 480 640
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 150, 255, cv2.THRESH_BINARY)

    crop_img = blackAndWhiteImage[camera_height-10:camera_height-9, 0:camera_width]
    mass = []
    for i in range(camera_width):
        mass.append(1) if crop_img[0, i] == 0 else mass.append(0)

    index_a = 0
    index_b = 0
    data = []
    left = 0
    right = 0

    for i in range(len(mass)):
        if mass[i] == 1:
            right = i
    for i in range(len(mass)):
        if mass[len(mass) - i - 1] == 1:
            left = i

    line_center = right + (640 - right - left) // 4

    print(f"Line left: {camera_width-left}     Line right: {right}     Car position: {line_center}")

    draw_dot(frame, 'right', right)
    draw_dot(frame, 'left', camera_width-left)
    draw_dot(frame, 'car', line_center)

    cv2.imshow('cam_0', frame)
    cv2.imshow('cam_0_black', blackAndWhiteImage)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
