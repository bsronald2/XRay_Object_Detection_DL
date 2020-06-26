import cv2
import numpy as np
from src.config import dim


def img_obj_detection(img_path):
    # Read Image
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, dsize=(dim[0], dim[1]))

    # Blur Image
    blur = cv2.medianBlur(img, 5)

    # 3D image to 2D
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    # Threshold - Get binary image
    thresh_hold = 37
    thresh = cv2.threshold(gray, thresh_hold, 255, cv2.THRESH_BINARY_INV)[1]

    # Find Contours
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1] # retrieve counts by version

    min_area = 50
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area: # If contour area is greater than min area draw over raw_img
            color = (36, 255, 12)
            thickness = 2
            cv2.drawContours(img, [c], -1, color, thickness)

    return img
