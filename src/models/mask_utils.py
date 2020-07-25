import numpy as np
import cv2
from datetime import datetime
import imageio
from src.config import labels, dim, hues_labels, n_classes


class Mask:

    def __init__(self, pred_path):
        self.pred_path = pred_path
        pass

    def color_masks(self, pred):
        mask = np.zeros(shape=(dim[0], dim[0], n_classes), dtype=np.uint8)

        for i, label in enumerate(labels):
            hue = np.full(shape=(dim[0], dim[1]), fill_value=hues_labels[label], dtype=np.uint8)
            sat = np.full(shape=(dim[0], dim[1]), fill_value=255, dtype=np.uint8)
            val = pred[:, :, i].astype(np.uint8)

            im_hsv = cv2.merge([hue, sat, val])
            im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
            mask = cv2.add(mask, im_rgb)

        return mask

    def color_binary_mask(self, pred):
        blank = np.zeros(shape=(dim[0], dim[1], 3), dtype=np.uint8)
        hue = np.full(shape=(dim[0], dim[1]), fill_value=50, dtype=np.uint8)
        sat = np.full(shape=(dim[0], dim[1]), fill_value=255, dtype=np.uint8)
        val = pred[:,:,0].astype(np.uint8)
        im_hsv = cv2.merge([hue, sat, val])
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)

        blank = cv2.add(blank, im_rgb)
        return blank


    def blending_2D_images(self, img, pred_mask, task='multi'):
        print(img.shape, pred_mask.shape)
        if task == 'multi':
            pred_color_mask = self.color_masks(pred_mask.squeeze() * 255.0)
        elif 'binary':
            pred_color_mask = self.color_binary_mask(pred_mask)
        stacked_img = np.stack((img.squeeze(),) * 3, axis=-1).astype(np.uint8)
        new_img = cv2.addWeighted(stacked_img, 1.0, pred_color_mask, 5.0, 0)
        time_stamp = datetime.timestamp(datetime.now())
        imageio.imwrite(f'{self.pred_path}/prediction_{time_stamp}.png', new_img)

    def blending_batch(self, X, y_pre):
        batch_size = X.shape[0]
        for index in range(batch_size):
            self.blending_2D_images(X[index], y_pre[index])