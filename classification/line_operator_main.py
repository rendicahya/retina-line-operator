import cv2
import numpy as np

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.multi_line_opr import cached_multi
from methods.single_line_opr import cached_line, subtract
from methods.window_average import cached_integral
from util.data_util import accuracy
from util.image_util import normalize_masked
from util.time import Time


def test(op, data):
    best_acc = 0
    best_thresh = 0
    size = 15
    time = Time()

    for thresh in range(1, 255):
        acc_list = []

        time.start(thresh)

        for path, img, mask, ground in data:
            img = 255 - img[:, :, 1]

            linestr = op(path, img, mask, size)
            bin = cv2.threshold(linestr, thresh, 255, cv2.THRESH_BINARY)[1]
            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)

        avg = np.average(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

        time.finish()
        print(f'{avg}')

    print(f'Best threshold: {best_thresh}')
    print(f'Best accuracy: {best_acc}')


def single(path, img, mask, size):
    window_avg = cached_integral(path, img, mask, size)
    line_img = cached_line(path, img, mask, size)
    linestr = subtract(line_img, window_avg, mask)

    return normalize_masked(linestr, mask)


def multi(path, img, mask, size):
    linestr = cached_multi(path, img, mask, size)

    return normalize_masked(linestr, mask)


if __name__ == '__main__':
    data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    test(multi, data)

    # cv2.imshow('Image', linestr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
