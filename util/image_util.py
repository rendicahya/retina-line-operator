import cv2
import numpy as np

from methods.single_line_opr import cached_line, subtract
from methods.window_average import cached_integral
from util.data_util import accuracy
from util.time import Time


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def normalize_masked(image, mask):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask)


def find_thresh_one(image, ground, mask):
    best_acc = 0
    best_thresh = 0
    best_image = None

    for t in range(1, 255):
        thresh, bin = cv2.threshold(image, t, 255, cv2.THRESH_BINARY)
        bin_fov = bin[mask == 255]
        ground_fov = ground[mask == 255]
        acc = accuracy(bin_fov, ground_fov)

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_image = bin

    return best_thresh, best_image, best_acc


def find_thresh_all(dataset):
    best_acc = 0
    best_thresh = 0

    for thresh in range(1, 255):
        acc_list = []

        for path, img, mask, ground in dataset:
            img = 255 - img[:, :, 1]
            size = 15
            time = Time()

            time.start(path)
            window_avg = cached_integral(path, img, mask, size)
            line_img = cached_line(path, img, mask, size)
            single_img = subtract(line_img, window_avg, mask)
            single_img = normalize_masked(single_img, mask)
            bin = cv2.threshold(single_img, thresh, 255, cv2.THRESH_BINARY)[1]
            acc = accuracy(bin, ground)

            acc_list.append(acc)
            time.finish()

        avg = np.average(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

    return best_thresh, best_acc
