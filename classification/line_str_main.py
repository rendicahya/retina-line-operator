import cv2

from dataset import DriveDatasetLoader
from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.single_line_opr import cached_line, subtract
from methods.window_average import cached_integral
from util.image_util import find_best_threshold
from util.time import Time
from util.print_color import *
from util.image_util import normalize_masked
from util.data_util import accuracy


def threshold_all(thresh):
    accuracies = []

    for path, img, mask, ground in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
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
        accuracies.append(acc)
        time.finish()

    print(accuracies)


def find_best():
    for img_id in range(1, 5):
        path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing_one(img_id)
        img = 255 - img[:, :, 1]
        size = 15
        time = Time()

        time.start(path)
        window_avg = cached_integral(path, img, mask, size)
        line_img = cached_line(path, img, mask, size)
        single_img = subtract(line_img, window_avg, mask)
        single_img = normalize_masked(single_img, mask)
        best_thresh, best_img = find_best_threshold(single_img, ground, mask)
        acc = accuracy(best_img, ground)
        time.finish()

        green(f'Best threshold: {best_thresh}')

        # cv2.imshow('Single Line Operator', single_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    threshold_all(60)
