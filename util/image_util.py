import cv2
import numpy as np

from util.data_util import accuracy


def clahe(img, clip_limit=2.0, tile_grid_size=(5, 5)):
    return cv2.createCLAHE(clip_limit, tile_grid_size).apply(img)


def gray_norm(img, mask=None):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask)


def subtract_line_str(line, window, mask):
    return cv2.subtract(line.astype(np.float64), window, None, mask)


def zero_one_norm(data):
    return cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)


def find_best_thresh(line_str, ground, mask):
    line_str = gray_norm(line_str, mask)
    ground_fov = ground[mask == 255]
    acc_list = []
    img_list = []

    for t in range(1, 255):
        bin = cv2.threshold(line_str, t, 255, cv2.THRESH_BINARY)[1]
        bin_fov = bin[mask == 255]
        acc = accuracy(bin_fov, ground_fov)

        acc_list.append(acc)
        img_list.append(bin)

    best = np.argmax(acc_list)

    return best + 1, img_list[best], acc_list[best]
