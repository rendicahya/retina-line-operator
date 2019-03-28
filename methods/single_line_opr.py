import multiprocessing as mp
import os.path
import pickle
import sys

import psutil

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods import window_average, line_factory
from util.image_util import *
from util.time import Time


def single(line, window, mask):
    line = line.astype(np.float64)

    return cv2.subtract(line, window, None, mask)


def cached_line(path, img, mask, size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/line-%s-%d.bin' % (cache_dir, os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_strength = pickle.load(binary_file)
    else:
        line_strength = line(img, mask, size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_strength, binary_file)

    binary_file.close()

    return line_strength


def line(img, mask, size):
    img = img.astype(np.int16)
    bool_mask = mask.astype(np.bool)
    lines, wings = line_factory.generate_lines(size)

    queue = mp.Queue()
    cpu_count = psutil.cpu_count()

    processes = [
        mp.Process(target=line_worker, args=(img, bool_mask, lines, queue, cpu_count, cpu_id))
        for cpu_id in range(cpu_count)]

    for p in processes:
        p.start()

    slices = [queue.get() for _ in processes]
    slices = sorted(slices, key=lambda s: s[0])
    slices = [piece[1] for piece in slices]

    for p in processes:
        p.join()

    return np.vstack(slices)


def line_worker(img, bool_mask, lines, queue, cpu_count, cpu_id):
    height, width = img.shape[:2]
    slice_height = height // cpu_count
    y_start = cpu_id * slice_height
    linestr = np.zeros((slice_height, width), np.int16)

    for Y in range(y_start, (cpu_id + 1) * slice_height):
        for X in range(width):
            if not bool_mask[Y, X]:
                continue

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
                    line_sum += img[y, x]

                if line_count == 0:
                    continue

                line_avg = line_sum / line_count

                if line_avg > max_line_avg:
                    max_line_avg = line_avg

            linestr[Y - y_start, X] = max_line_avg

    queue.put((cpu_id, linestr))


def cache_all():
    time = Time()

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training():
        img = 255 - img[:, :, 1]

        for size in range(1, 16, 2):
            time.start('%s [%d]' % (path, size))
            window_avg = window_average.cached_integral(path, img, mask, size)
            # cached_line(path, img, mask, size)
            time.finish()


def main():
    path, img, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    img = 255 - img[:, :, 1]
    size = 15
    time = Time()

    time.start('Window average')
    window_avg = window_average.cached_integral(path, img, mask, size)
    time.finish()

    time.start('Single')
    line_img = cached_line(path, img, mask, size)
    single_img = single(line_img, window_avg, mask)
    time.finish()

    # time.start('Single scale + wing')
    # single_scale_wing = single(img, mask, window_avg, size)
    # time.finish()

    # time.start('Find best threshold')
    # best_single_thresh, best_single = find_best_threshold(single_img, mask, ground_truth)
    # time.finish()

    # time.start('Multi scale')
    # multi_scale = cached_multi(path, img, mask, size)
    # time.finish()

    # time.start('Find best multi scale')
    # best_multi_thresh, best_multi = find_best_threshold(multi_scale, mask, ground_truth)
    # time.finish()

    # green('Best single scale threshold: %d' % best_single_thresh)
    # green('Best multi scale threshold: %d' % best_multi_thresh)

    cv2.imshow('Image', img)
    # cv2.imshow('Window average', normalize_masked(window_avg, mask))
    cv2.imshow('Single', normalize_masked(single_img, mask))
    # cv2.imshow('Single + wing', normalize_masked(255 - single_scale_wing, mask))
    # cv2.imshow('Single best', 255 - normalize_masked(best_single, mask))
    # cv2.imshow('Multi', normalize_masked(multi_scale, mask))
    # cv2.imshow('Best multi', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Multi histeq', cv2.equalizeHist(multi_scale))
    # cv2.imshow('Ground truth', ground_truth)
    # cv2.imshow('Best binary', binary)
    # cv2.imshow('Multi', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
