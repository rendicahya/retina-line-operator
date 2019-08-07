import cv2
import numpy as np

from methods.optic_disk import cached_disk_norm
from methods.proposed import proposed_norm
from util.data_util import auc_score, accuracy
from util.image_util import find_best_thresh, gray_norm


def find_best_acc(op, data, size):
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


def find_best_acc_each(op, data, size):
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


def get_accuracy(op, data, thresh, size):
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


def find_best_acc_optic(op, thresh, data, size):
    avg_acc_list = []

    for disk_thresh in range(1, 255):
        acc_list = []

        for path, img, mask, ground in data:
            img = 255 - img[:, :, 1]

            disk = cached_disk_norm(path, img, mask, size)
            disk = cv2.threshold(disk, disk_thresh, 255, cv2.THRESH_BINARY)[1]
            disk = cv2.erode(disk, np.ones((3, 3), np.uint8), iterations=1)

            line_str = op(path, img, mask, size)
            line_str[disk == 255] = line_str[mask == 255].min()
            line_str = gray_norm(line_str, mask)
            bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]

            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)

        avg_acc = np.mean(acc_list)
        avg_acc_list.append(avg_acc)

    best = np.argmax(avg_acc_list)

    return best + 1, avg_acc_list[best]


def find_best_acc_optic_each(op, data, size):
    acc_list = []
    auc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]
        line_str = op(path, img, mask, size)
        line_str_norm = gray_norm(line_str, mask)
        bin = find_best_thresh(line_str_norm, ground, mask)[1]
        temp_acc = []

        for disk_thresh in range(1, 255):
            optic = cached_disk_norm(path, img, mask, size)
            optic = cv2.threshold(optic, disk_thresh, 255, cv2.THRESH_BINARY)[1]
            optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)
            bin_subtract = bin.copy()

            bin_subtract[optic == 255] = 0
            acc = accuracy(ground, bin_subtract)

            temp_acc.append(acc)

        best_acc = np.max(temp_acc)

        best_disk_thresh = np.argmax(temp_acc) + 1
        optic = cached_disk_norm(path, img, mask, size)
        optic = cv2.threshold(optic, best_disk_thresh, 255, cv2.THRESH_BINARY)[1]
        optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)
        line_str[optic == 255] = line_str.min()
        auc = auc_score(ground, line_str, mask)

        acc_list.append(best_acc)
        auc_list.append(auc)

    return np.mean(acc_list), np.mean(auc_list)


def find_best_acc_proposed(op, thresh, optic_thresh, data):
    best_acc = 0
    best_thresh = 0
    size = 15

    for proposed_thresh in range(1, 255):
        acc_list = []

        for path, img, mask, ground in data:
            img = 255 - img[:, :, 1]
            line_str = op(path, img, mask, size)
            bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]

            optic = cached_disk_norm(path, img, mask, size)
            optic = cv2.threshold(optic, optic_thresh, 255, cv2.THRESH_BINARY)[1]
            optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)

            min_window = proposed_norm(path, img, mask, size)
            min_window = 255 - cv2.threshold(min_window, proposed_thresh, 255, cv2.THRESH_BINARY)[1]
            min_window[mask == 0] = 0
            bin[min_window == 255] = 0
            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)

        avg = np.mean(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

    return best_thresh, best_acc


def get_accuracy_optic(op, data, thresh, disk_thresh):
    size = 15
    acc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]
        line_str = op(path, img, mask, size)
        line_str = gray_norm(line_str, mask)
        bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]

        disk = cached_disk_norm(path, img, mask, size)
        disk = cv2.threshold(disk, disk_thresh, 255, cv2.THRESH_BINARY)[1]
        disk = cv2.erode(disk, np.ones((3, 3), np.uint8), iterations=1)
        bin[disk == 255] = 0
        bin_fov = bin[mask == 255]
        ground_fov = ground[mask == 255]
        acc = accuracy(bin_fov, ground_fov)

        acc_list.append(acc)

    return np.mean(acc_list)


def get_accuracy_proposed(op, data, thresh, proposed_thresh):
    size = 15
    acc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]
        line_str = op(path, img, mask, size)
        bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]
        min_window = proposed_norm(path, img, mask, size)
        min_window = 255 - cv2.threshold(min_window, proposed_thresh, 255, cv2.THRESH_BINARY)[1]
        min_window[mask == 0] = 0
        bin[min_window == 255] = 0
        bin_fov = bin[mask == 255]
        ground_fov = ground[mask == 255]
        acc = accuracy(bin_fov, ground_fov)

        acc_list.append(acc)

    return np.mean(acc_list)
