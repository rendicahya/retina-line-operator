import cv2
import numpy as np

from util.data_util import accuracy


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def normalize_masked(img, mask):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask)


def find_best_thresh(img, ground, mask):
    best_acc = 0
    best_thresh = 0
    best_img = None

    for t in range(1, 255):
        thresh, bin = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        bin_fov = bin[mask == 255]
        ground_fov = ground[mask == 255]
        acc = accuracy(bin_fov, ground_fov)

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_img = bin

    return best_thresh, best_img, best_acc


def subtract_masked(line, window, mask):
    return cv2.subtract(line.astype(np.float64), window, None, mask)
