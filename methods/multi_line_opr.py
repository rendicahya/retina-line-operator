import os.path
import pickle

import cv2
import numpy as np

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.image_util import norm_masked, subtract_masked
from util.timer import Timer


def cached_multi_norm(path, img, mask, size):
    line_str = cached_multi(path, img, mask, size)

    return norm_masked(line_str, mask)


def cached_multi(path, img, mask, size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/multi-%s-%d.bin' % (cache_dir, os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_str = pickle.load(binary_file)
    else:
        window = cached_integral(path, img, mask, size)
        line_str = [subtract_masked(cached_statistics(path, img, mask, size)['max'], window, mask)
                    for size in range(1, size + 1, 2)]
        line_str = np.average(np.stack(line_str), axis=0)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_str, binary_file)

    binary_file.close()

    return line_str


def save_cache():
    time = Timer()
    size = 15

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        img = 255 - img[:, :, 1]

        time.start(path)
        cached_multi(path, img, mask, size)
        time.stop()


def main():
    path, img, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    img = 255 - img[:, :, 1]
    size = 15
    timer = Timer()

    timer.start('Multi scale')
    multi_scale = cached_multi_norm(path, img, mask, size)
    timer.stop()

    # timer.start('Find best multi scale')
    # best_multi_thresh, best_multi = find_best_threshold(multi_scale, mask, ground_truth)
    # timer.finish()

    cv2.imshow('Image', img)
    cv2.imshow('Multi', multi_scale)
    # cv2.imshow('Best multi', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Ground truth', ground_truth)
    # cv2.imshow('Best binary', binary)
    # cv2.imshow('Multi', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    save_cache()
