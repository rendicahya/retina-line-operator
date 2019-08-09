import cv2
import numpy as np

from methods.optic_disk import cached_optic_norm
from methods.proposed import proposed_norm
from util.data_util import accuracy
from util.image_util import find_best_thresh
from util.image_util import gray_norm


def proposed_train(op, thresh, optic_thresh, data, size):
    avg_acc_list = []

    for proposed_thresh in range(1, 255):
        acc_list = []

        for path, img, mask, ground in data:
            img = 255 - img[:, :, 1]

            line_str = op(path, img, mask, size)
            line_str = gray_norm(line_str, mask)
            bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]

            optic = cached_optic_norm(path, img, mask, size)
            optic = cv2.threshold(optic, optic_thresh, 255, cv2.THRESH_BINARY)[1]
            optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)
            bin[optic == 255] = 0

            min_window = proposed_norm(path, img, mask, size)
            min_window = cv2.threshold(min_window, proposed_thresh, 255, cv2.THRESH_BINARY)[1]
            bin[min_window == 255] = 0

            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)

            # cv2.imshow('Before proposed', bin)
            # cv2.imshow('Line', line_str)
            # cv2.imshow('Min-window', min_window)
            # cv2.imshow('Proposed', bin)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # return

        avg_acc = np.mean(acc_list)

        avg_acc_list.append(avg_acc)

    best = np.argmax(avg_acc_list)

    return best + 1, avg_acc_list[best]


def proposed_test(op, data, thresh, optic_thresh, proposed_thresh):
    size = 15
    acc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]

        line_str = op(path, img, mask, size)
        line_str = gray_norm(line_str, mask)
        bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]

        optic = cached_optic_norm(path, img, mask, size)
        optic = cv2.threshold(optic, optic_thresh, 255, cv2.THRESH_BINARY)[1]
        optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)
        bin[optic == 255] = 0

        min_window = proposed_norm(path, img, mask, size)
        min_window = cv2.threshold(min_window, proposed_thresh, 255, cv2.THRESH_BINARY)[1]
        bin[min_window == 255] = 0

        bin_fov = bin[mask == 255]
        ground_fov = ground[mask == 255]
        acc = accuracy(bin_fov, ground_fov)

        acc_list.append(acc)

    return np.mean(acc_list)


def proposed_test_each(op, data, size):
    acc_list = []
    auc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]

        line_str = op(path, img, mask, size)
        line_str_norm = gray_norm(line_str, mask)
        bin = find_best_thresh(line_str_norm, ground, mask)[1]

        temp_acc = []

        for optic_thresh in range(1, 255):
            optic = cached_optic_norm(path, img, mask, size)
            optic = cv2.threshold(optic, optic_thresh, 255, cv2.THRESH_BINARY)[1]
            optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)

            bin_subtract = bin.copy()
            bin_subtract[optic == 255] = 0
            acc = accuracy(ground, bin_subtract)

            temp_acc.append(acc)

        best_optic_thresh = np.argmax(temp_acc) + 1

        optic = cached_optic_norm(path, img, mask, size)
        optic = cv2.threshold(optic, best_optic_thresh, 255, cv2.THRESH_BINARY)[1]
        optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)

        temp_acc = []

        for proposed_thresh in range(1, 255):
            min_window = proposed_norm(path, img, mask, size)
            min_window = cv2.threshold(min_window, proposed_thresh, 255, cv2.THRESH_BINARY)[1]

            bin_subtract = bin.copy()
            bin_subtract[optic == 255] = 0
            bin_subtract[min_window == 255] = 0

            acc = accuracy(ground, bin_subtract)

            temp_acc.append(acc)

        best_acc = np.max(temp_acc)

        acc_list.append(best_acc)

    return np.mean(acc_list)
