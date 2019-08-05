import cv2
import numpy as np

from util.data_util import auc_score, accuracy


def calc_auc(data, op):
    size = 15
    auc_list = []

    for path, img, mask, ground in data:
        line_str = op(path, img, mask, size)
        auc = auc_score(ground, line_str, mask)

        auc_list.append(auc)

    return np.mean(auc_list)


def find_best_acc_train_data(op, data):
    best_acc = 0
    best_thresh = 0
    size = 15

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

        avg = np.mean(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

    return best_thresh, best_acc
