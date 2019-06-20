import multiprocessing as mp
import os.path
import pickle
import sys

import numpy as np
import psutil

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods import line_factory
from util.image_util import *
from util.time import Time


def cached_basic(path, image, mask, window_size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/minmax-%s-%d.bin' % (cache_dir, os.path.basename(path), window_size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_strength = pickle.load(binary_file)
    else:
        line_strength = basic(image, mask, window_size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_strength, binary_file)

    binary_file.close()

    return line_strength


def basic(image, mask, window_size):
    image = image.astype(np.int16)
    bool_mask = mask.astype(np.bool)
    lines, wings = line_factory.generate_lines(window_size)

    queue = mp.Queue()
    cpu_count = psutil.cpu_count()

    processes = [
        mp.Process(target=basic_worker, args=(image, bool_mask, lines, queue, cpu_count, cpu_id))
        for cpu_id in range(cpu_count)]

    for p in processes:
        p.start()

    slices = [queue.get() for _ in processes]
    slices = sorted(slices, key=lambda slice: slice[0])
    slices = [piece[1] for piece in slices]

    for p in processes:
        p.join()

    return np.vstack(slices)


def basic_worker(image, bool_mask, lines, queue, cpu_count, cpu_id):
    height, width = image.shape[:2]
    slice_height = height // cpu_count
    y_start = cpu_id * slice_height
    minmax = np.zeros((slice_height, width), np.int16)

    for Y in range(y_start, (cpu_id + 1) * slice_height):
        for X in range(width):
            if not bool_mask[Y, X]:
                continue

            min_line_avg = sys.maxsize
            max_line_avg = -sys.maxsize - 1

            for angle, line in enumerate(lines):
                line_count = 0
                line_sum = 0

                for pixel in line:
                    x = X + pixel[0]
                    y = Y + pixel[1]

                    if x < 0 or x >= width or y < 0 or y >= height or not bool_mask[y, x]:
                        continue

                    line_count += 1
                    line_sum += image[y, x]

                if line_count == 0:
                    continue

                line_avg = line_sum / line_count

                if line_avg < min_line_avg:
                    min_line_avg = line_avg

                if line_avg > max_line_avg:
                    max_line_avg = line_avg

            minmax[Y - y_start, X] = max_line_avg - min_line_avg

    queue.put((cpu_id, minmax))


def cache_all():
    time = Time()

    for path, image, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        image = 255 - image[:, :, 1]

        for size in range(1, 26, 2):
            time.start('%s/%d' % (path, size))
            cached_basic(path, image, mask, size)
            time.finish()


def main():
    path, image, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    image = 255 - image[:, :, 1]
    size = 15
    time = Time()

    time.start('Minmax')
    single_scale = basic(image, mask, size)
    time.finish()

    # time.start('Single scale + wing')
    # single_scale_wing = single(image, mask, window_avg, size)
    # time.finish()

    # time.start('Find best threshold')
    # best_single_thresh, best_single = find_best_threshold(single_scale, mask, ground_truth)
    # time.finish()

    # time.start('Multi scale')
    # multi_scale = multi(path, image, mask, size)
    # time.finish()

    # time.start('Find best multi scale')
    # best_multi_thresh, best_multi = find_best_threshold(multi_scale, mask, ground_truth)
    # time.finish()

    # green('Best single scale threshold: %d' % best_single_thresh)
    # green('Best multi scale threshold: %d' % best_multi_thresh)

    cv2.imshow('Image', image)
    cv2.imshow('Minmax', normalize_masked(single_scale, mask))
    # cv2.imshow('Single scale + wing', normalize_masked(255 - single_scale_wing, mask))
    # cv2.imshow('Single scale best', 255 - normalize_masked(best_single, mask))
    # cv2.imshow('Multi scale', normalize_masked(multi_scale, mask))
    # cv2.imshow('Best multi scale', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Multi scale histeq', cv2.equalizeHist(multi_scale))
    # cv2.imshow('Ground truth', ground_truth)
    # cv2.imshow('Best binary', binary)
    # cv2.imshow('Multi scale', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
