import cv2
import numpy as np

from util.data_util import accuracy


def grayscale_norm(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def gray_norm_masked(img, mask):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask)


def subtract_masked(line, window, mask):
    return cv2.subtract(line.astype(np.float64), window, None, mask)


def find_best_thresh(line_str, ground, mask):
    line_str = gray_norm_masked(line_str, mask)
    ground_fov = ground[mask == 255]
    acc_list = []
    img_list = []

    for t in range(1, 255):
        bin = cv2.threshold(line_str, t, 255, cv2.THRESH_BINARY)[1]
        bin_fov = bin[mask == 255]

        acc_list.append(accuracy(bin_fov, ground_fov))
        img_list.append(bin)

    best = np.argmax(acc_list)

    return best + 1, img_list[best], acc_list[best]
