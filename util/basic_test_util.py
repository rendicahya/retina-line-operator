import cv2
import numpy as np

from util.data_util import auc_score, accuracy
from util.image_util import find_best_thresh


def basic_train(op, data, size):
    avg_acc_list = []

    for thresh in range(1, 255):
        acc_list = []

        for path, img, mask, ground in data:
            img = 255 - img[:, :, 1]
            line_str = op(path, img, mask, size)
            bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]
            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)

        avg_acc = np.mean(acc_list)
        avg_acc_list.append(avg_acc)

    best = np.argmax(avg_acc_list)

    return best + 1, avg_acc_list[best]


def basic_test_each(op, data, size):
    acc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]
        line_str = op(path, img, mask, size)
        thresh, bin, acc = find_best_thresh(line_str, ground, mask)

        acc_list.append(acc)

    return np.mean(acc_list)


def calc_auc(data, op, size):
    auc_list = []

    for path, img, mask, ground in data:
        line_str = op(path, img, mask, size)
        auc = auc_score(ground, line_str, mask)

        auc_list.append(auc)

    return np.mean(auc_list)


def basic_get_acc(op, data, thresh, size):
    acc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]
        line_str = op(path, img, mask, size)
        bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]
        bin_fov = bin[mask == 255]
        ground_fov = ground[mask == 255]
        acc = accuracy(bin_fov, ground_fov)

        acc_list.append(acc)

    return np.mean(acc_list)


