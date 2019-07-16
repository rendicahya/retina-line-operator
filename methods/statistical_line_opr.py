import multiprocessing as mp
import os.path
import pickle

import cv2
import numpy as np
import psutil

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods import window_average
from methods.line_factory import generate_lines
from util.image_util import subtract_masked, normalize_masked, find_best_thresh
from util.timer import Timer


def cached_statistics(path, img, mask, size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/statistical-%s-%d.bin' % (cache_dir, os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_strength = pickle.load(binary_file)
    else:
        line_strength = statistics(img, mask, size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_strength, binary_file)

    binary_file.close()

    return line_strength


def statistics(img, mask, size):
    bool_mask = mask.astype(np.bool)
    lines, wings = generate_lines(size)

    queue = mp.Queue()
    cpu_count = psutil.cpu_count()

    processes = [
        mp.Process(target=statistics_worker, args=(img, bool_mask, lines, queue, cpu_count, cpu_id))
        for cpu_id in range(cpu_count)]

    for p in processes:
        p.start()

    slices = [queue.get() for _ in processes]
    slices = sorted(slices, key=lambda s: s[0])
    slices = [piece[1] for piece in slices]

    for p in processes:
        p.join()

    stack = np.vstack(slices)

    return {'max': stack[..., 0],
            'min': stack[..., 1],
            'mean': stack[..., 2],
            'std': stack[..., 3]}


def statistics_worker(img, bool_mask, lines, queue, cpu_count, cpu_id):
    height, width = img.shape[:2]
    slice_height = height // cpu_count
    y_start = cpu_id * slice_height
    stat = np.zeros((slice_height, width, 4), np.float64)
    temp_line_str = np.empty(len(lines), np.float64)

    for Y in range(y_start, (cpu_id + 1) * slice_height):
        for X in range(width):
            if not bool_mask[Y, X]:
                continue

            for angle, line in enumerate(lines):
                pixel_count = 0
                pixel_sum = 0

                for pixel in line:
                    x = X + pixel[0]
                    y = Y + pixel[1]

                    if x < 0 or x >= width or y < 0 or y >= height or not bool_mask[y, x]:
                        continue

                    pixel_count += 1
                    pixel_sum += img[y, x]

                temp_line_str[angle] = pixel_sum / pixel_count

            stat[Y - y_start, X] = [np.max(temp_line_str),
                                    np.min(temp_line_str),
                                    np.mean(temp_line_str),
                                    np.std(temp_line_str)]

    queue.put((cpu_id, stat))


def save_cache():
    time = Timer()
    size = 15

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training():
        img = 255 - img[:, :, 1]

        time.start(path)
        cached_statistics(path, img, mask, size)
        time.stop()


def main():
    path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    img = 255 - img[:, :, 1]
    size = 15
    timer = Timer()

    timer.start('Window average')
    window_avg = window_average.cached_integral(path, img, mask, size)
    timer.stop()

    timer.start('Single')
    statistics = cached_statistics(path, img, mask, size)
    maxi = statistics['max']
    mini = statistics['min']
    mean = statistics['mean']
    std = statistics['std']
    timer.stop()

    # timer.start('Single scale + wing')
    # single_scale_wing = single(img, mask, window_avg, size)
    # timer.finish()

    # timer.start('Find best threshold')
    # best_single_thresh, best_single = find_best_threshold(result, mask, ground)
    # timer.finish()

    # timer.start('Multi scale')
    # multi_scale = cached_multi(path, img, mask, size)
    # timer.finish()

    # timer.start('Find best multi scale')
    # best_multi_thresh, best_multi = find_best_threshold(multi_scale, mask, ground)
    # timer.finish()

    # green('Best single scale threshold: %d' % best_single_thresh)
    # green('Best multi scale threshold: %d' % best_multi_thresh)

    cv2.imshow('Image', img)
    # cv2.imshow('Window average', normalize_masked(window_avg, mask))
    cv2.imshow('Max', normalize_masked(maxi, mask))
    cv2.imshow('Max-window', normalize_masked(subtract_masked(maxi, window_avg, mask), mask))
    cv2.imshow('Max-window-thresh',
               normalize_masked(find_best_thresh(subtract_masked(maxi, window_avg, mask), ground, mask)[1], mask))
    cv2.imshow('Min', normalize_masked(mini, mask))
    cv2.imshow('Min-window', normalize_masked(subtract_masked(mini, window_avg, mask), mask))
    cv2.imshow('Mean', normalize_masked(mean, mask))
    cv2.imshow('Mean-window', normalize_masked(subtract_masked(mean, window_avg, mask), mask))
    cv2.imshow('Std', normalize_masked(std, mask))
    cv2.imshow('Std-window', normalize_masked(subtract_masked(std, window_avg, mask), mask))
    # cv2.imshow('Single + wing', normalize_masked(255 - single_scale_wing, mask))
    # cv2.imshow('Single best', 255 - normalize_masked(best_single, mask))
    # cv2.imshow('Multi', normalize_masked(multi_scale, mask))
    # cv2.imshow('Best multi', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Multi histeq', cv2.equalizeHist(multi_scale))
    # cv2.imshow('Ground truth', ground)
    # cv2.imshow('Best binary', binary)
    # cv2.imshow('Multi', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
