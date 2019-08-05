import cv2
import numpy as np

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.multi_line_opr import cached_multi_norm
from methods.optic_disk import cached_disk_norm
from methods.proposed import proposed_norm
from methods.single_line_opr import cached_single_norm
from util.data_util import accuracy
from util.image_util import find_best_thresh
from util.print_color import *
from util.test_util import find_best_acc_train_data
from util.timer import Timer


def get_accuracy(op, data, thresh):
    size = 15
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


def find_best_acc_disk(op, thresh, data):
    best_acc = 0
    best_thresh = 0
    size = 15

    for disk_thresh in range(1, 255):
        acc_list = []

        for path, img, mask, ground in data:
            img = 255 - img[:, :, 1]
            line_str = op(path, img, mask, size)
            bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]

            disk = cached_disk_norm(path, img, mask, size)
            disk = cv2.threshold(disk, disk_thresh, 255, cv2.THRESH_BINARY)[1]
            disk = cv2.erode(disk, np.ones((3, 3), np.uint8), iterations=1)
            bin[disk == 255] = 0
            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)

        avg = np.mean(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

    return best_thresh, best_acc


def find_best_acc_proposed(op, thresh, data):
    best_acc = 0
    best_thresh = 0
    size = 15

    for proposed_thresh in range(1, 255):
        acc_list = []

        print(proposed_thresh)

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

        avg = np.mean(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

    return best_thresh, best_acc


def get_accuracy_disk(op, data, thresh, disk_thresh):
    size = 15
    acc_list = []

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]
        line_str = op(path, img, mask, size)
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


def find_best_acc_avg_each(op, data):
    acc_list = []
    size = 15

    for path, img, mask, ground in data:
        img = 255 - img[:, :, 1]
        line_str = op(path, img, mask, size)
        acc = find_best_thresh(line_str, ground, mask)[2]

        acc_list.append(acc)

    print(np.mean(acc_list))
    print(np.max(acc_list))


def test_line():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()
    op = cached_single_norm
    # op = cached_multi_norm
    timer = Timer()

    timer.start('Train')
    thresh, train_acc = find_best_acc_train_data(op, train_data)
    timer.stop()

    timer.start('Test')
    test_acc = get_accuracy(op, test_data, thresh)
    timer.stop()

    blue(f'Threshold: {thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')

    # cv2.imshow('Image', linestr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def test_optic():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()
    # op = cached_single_norm
    op = cached_multi_norm
    thresh = 119
    timer = Timer()

    timer.start('Train')
    disk_thresh, train_acc = find_best_acc_disk(op, thresh, train_data)
    timer.stop()

    timer.start('Test')
    test_acc = get_accuracy_disk(op, test_data, thresh, disk_thresh)
    timer.stop()

    blue(f'Disk threshold: {disk_thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')


def test_proposed():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()
    op = cached_single_norm
    # op = cached_multi_norm
    thresh = 64
    timer = Timer()

    timer.start('Train')
    proposed_thresh, train_acc = find_best_acc_proposed(op, thresh, train_data)
    timer.stop()

    timer.start('Test')
    test_acc = get_accuracy_proposed(op, test_data, thresh, proposed_thresh)
    timer.stop()

    blue(f'Proposed threshold: {proposed_thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')


if __name__ == '__main__':
    test_proposed()
