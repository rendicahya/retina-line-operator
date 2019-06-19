import cv2

from util.data_util import accuracy


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def normalize_masked(image, mask):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U, mask)


def find_best_threshold(image, ground, mask):
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

    return best_thresh, best_image
