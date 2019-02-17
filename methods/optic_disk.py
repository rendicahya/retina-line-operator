import multiprocessing as mp
import os.path
import pickle
import sys

import cv2
import numpy as np
import psutil

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods import line_factory
from util.image_util import normalize_masked
from util.time import Time


def cached_basic(path, image, mask, window_size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/optdisk-%s-%d.bin' % (cache_dir, os.path.basename(path), window_size)

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
        mp.Process(target=basic_worker, args=(image, bool_mask, lines, wings, queue, cpu_count, cpu_id))
        for cpu_id in range(cpu_count)]

    for p in processes:
        p.start()

    slices = [queue.get() for _ in processes]
    slices = sorted(slices, key=lambda slice: slice[0])
    slices = [piece[1] for piece in slices]

    for p in processes:
        p.join()

    result = np.vstack(slices)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    result[mask == 0] = 0

    return result


def basic_worker(image, bool_mask, lines, wings, queue, cpu_count, cpu_id):
    height, width = image.shape[:2]
    slice_height = height // cpu_count
    y_start = cpu_id * slice_height
    linestr = np.zeros((slice_height, width), np.int16)

    for Y in range(y_start, (cpu_id + 1) * slice_height):
        for X in range(width):
            if not bool_mask[Y, X]:
                continue

            max_line_avg = -sys.maxsize - 1
            max_angle = 0

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

                if line_avg > max_line_avg:
                    max_line_avg = line_avg
                    max_angle = angle

            wing = wings[max_angle]
            p = Y + wing[0][1], X + wing[0][0]
            q = Y + wing[1][1], X + wing[1][0]
            linestr[Y - y_start, X] = abs(image[p] - image[q])

    queue.put((cpu_id, linestr))


def cache_all():
    time = Time()

    for path, image, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        image = 255 - image[:, :, 1]

        for size in range(15, 26, 2):
            time.start('%s/%d' % (path, size))
            basic(image, mask, size)
            time.finish()


def main():
    path, image, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(2)
    image = 255 - image[:, :, 1]
    size = 15
    time = Time()

    time.start('Optic disk')
    optdisk = cached_basic(path, image, mask, size)
    time.finish()

    optdisk = normalize_masked(optdisk, mask)
    th, optdisk = cv2.threshold(optdisk, 75, 255, cv2.THRESH_BINARY)
    optdisk = cv2.erode(optdisk, np.ones((3, 3), np.uint8), iterations=1)
    image[optdisk == 255] = 255

    cv2.imshow('Image', image)
    cv2.imshow('Optic disk', optdisk)
    cv2.imshow('Ground truth', ground_truth)
    cv2.imwrite('C:/Users/Rendicahya/Desktop/optdisk.jpg', optdisk)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
