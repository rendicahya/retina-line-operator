import os.path
import pickle

import cv2
import numpy as np

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.single_line_opr import subtract, cached_line
from methods.window_average import cached_integral
from util.image_util import normalize_masked
from util.time import Time


def cached_multi_norm(path, img, mask, size):
    linestr = cached_multi(path, img, mask, size)

    return normalize_masked(linestr, mask)


def cached_multi(path, img, mask, size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/multi-%s-%d.bin' % (cache_dir, os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_strength = pickle.load(binary_file)
    else:
        line_strength = multi(path, img, mask, size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_strength, binary_file)

    binary_file.close()

    return line_strength


def multi(path, img, mask, size):
    window = cached_integral(path, img, mask, size)
    line_str = [subtract(cached_line(path, img, mask, size), window, mask)
                for size in range(1, size + 1, 2)]

    return np.average(np.stack(line_str), axis=0)


def save_cache():
    time = Time()

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        img = 255 - img[:, :, 1]

        time.start(path)
        cached_multi(path, img, mask, 15)
        time.finish()


def main():
    path, img, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    img = 255 - img[:, :, 1]
    size = 15
    time = Time()

    time.start('Multi scale')
    multi_scale = cached_multi_norm(path, img, mask, size)
    time.finish()

    # time.start('Find best multi scale')
    # best_multi_thresh, best_multi = find_best_threshold(multi_scale, mask, ground_truth)
    # time.finish()

    # green('Best single scale threshold: %d' % best_single_thresh)
    # green('Best multi scale threshold: %d' % best_multi_thresh)

    cv2.imshow('Image', img)
    cv2.imshow('Multi', multi_scale)
    # cv2.imshow('Best multi', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Multi histeq', cv2.equalizeHist(multi_scale))
    # cv2.imshow('Ground truth', ground_truth)
    # cv2.imshow('Best binary', binary)
    # cv2.imshow('Multi', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
