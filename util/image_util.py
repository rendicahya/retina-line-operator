import cv2
import numpy as np

from util.data_util import accuracy


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def norm_masked(img, mask):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask)


def find_best_thresh(img, ground, mask):
    ground_fov = ground[mask == 255]
    thresh_list = []
    acc_list = []
    img_list = []

    for t in range(1, 255):
        bin = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)[1]
        bin_fov = bin[mask == 255]

        thresh_list.append(t)
        acc_list.append(accuracy(bin_fov, ground_fov))
        img_list.append(bin)

    best = np.argmax(acc_list)

    return thresh_list[best], img_list[best], acc_list[best]


def subtract_masked(line, window, mask):
    return cv2.subtract(line.astype(np.float64), window, None, mask)
