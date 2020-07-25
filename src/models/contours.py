import cv2
import numpy as np
from src.config import dim
from src.models.metrics import dice_coef


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
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # retrieve counts by version

    min_area = 50
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:  # If contour area is greater than min area draw over raw_img
            color = (36, 255, 12)
            thickness = 2
            cv2.drawContours(img, [c], -1, color, thickness)

    return img


def predict_contours_batch(data_generator, mask):
    metrics = []
    for X, y_exp in data_generator.get_iter():
        batch_shape = X.shape[0]
        for i in range(batch_shape):
            # Find contours and create mask
            thresh = threshold(np.stack((X[i].squeeze(),) * 3, axis=-1).astype(np.uint8))
            y_pred = morphology_operation(thresh)
            y_pred = np.expand_dims(y_pred, axis=2)
            # save predict mask over original img
            mask.blending_2D_images(X[i], y_pred, 'binary')
            # Calculate dice
            dice_result = dice_coef(y_exp[i], y_pred)
            metrics.append(dice_result)

    result = np.mean(metrics)
    print('Dice Results:', result)
    return result


def threshold(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur Image
    blur = cv2.medianBlur(grey, 5)
    # Threshold - Get binary image
    thresh_hold = 37
    thresh = cv2.threshold(blur, thresh_hold, 255, cv2.THRESH_BINARY_INV)[1]
    # Find Contours
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]  # retrieve counts by version
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    max_cnt_area = 0
    if len(cnts) > 0:
        max_cnt_area = cv2.contourArea(cnts[0])
    print('Max Count Area', max_cnt_area)
    if max_cnt_area > 10000:
        _, thresh = cv2.threshold(grey, 37, 255, cv2.THRESH_BINARY_INV)
    else:
        thresh = np.ones(shape=(512, 512), dtype=np.uint8) * 255
    return thresh


def morphology_operation(thresh):
    mask = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask
