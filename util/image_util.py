import cv2
import numpy as np

from util.data_util import accuracy


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def normalize_masked(image, mask):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask)


def find_best_threshold(image, mask, ground_truth):
    best_correct = 0
    best_threshold = 0
    best_image = None

    for t in range(1, 255):
        threshold, binary = cv2.threshold(image, t, 255, cv2.THRESH_BINARY)
        binary_array = binary[mask == 255]
        ground_truth_array = ground_truth[mask == 255]
        correct = np.sum(binary_array == ground_truth_array)

        if correct > best_correct:
            best_correct = correct
            best_threshold = threshold
            best_image = binary

    return best_threshold, best_image


def find_best_threshold2(image, mask, gtruth):
    best_acc = 0
    best_thresh = 0
    best_image = None

    for t in range(1, 255):
        thresh, bin = cv2.threshold(image, t, 255, cv2.THRESH_BINARY)
        bin_fov = bin[mask == 255]
        gtruth_fov = gtruth[mask == 255]
        acc = accuracy(bin_fov, gtruth_fov)

        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            best_image = bin

    return best_thresh, best_image
