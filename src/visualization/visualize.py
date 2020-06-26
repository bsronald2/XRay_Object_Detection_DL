import numpy as np
import cv2

# Helper to identify HSV parameters of XRay images
def nothing():
    pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    # Image to detect
    img_path = '/home/ronald/PycharmProjects/x-ray-deep-learning/X-ray_Object_Detection/data/raw/multi/images/train/B0046_0100.png'

    # Read in image
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=(256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking") # 0
    l_s = cv2.getTrackbarPos("LS", "Tracking") # 0
    l_v = cv2.getTrackbarPos("LV", "Tracking") # 0

    u_h = cv2.getTrackbarPos("UH", "Tracking") # 0
    u_s = cv2.getTrackbarPos("US", "Tracking") # 0
    u_v = cv2.getTrackbarPos("UV", "Tracking") # 38

    lb = np.array([l_h, l_s, l_h])
    ub = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lb, ub)

    res = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('frame', img)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()