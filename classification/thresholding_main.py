import cv2
import numpy as np

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.single_line_opr import cached_single_norm
from methods.multi_line_opr import cached_multi_norm
from util.data_util import accuracy
from util.timer import Timer
from util.image_util import find_best_thresh


def find_best_acc_avg_all(op, data):
    best_acc = 0
    best_thresh = 0
    size = 15
    timer = Timer()

    for thresh in range(1, 255):
        acc_list = []

        timer.start(thresh)

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

        timer.stop()
        print(avg)

    print(f'Best threshold: {best_thresh}')
    print(f'Best accuracy: {best_acc}')


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


def main():
    data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()

    find_best_acc_avg_all(cached_multi_norm, data)

    # cv2.imshow('Image', linestr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
