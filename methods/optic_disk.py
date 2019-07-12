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
from util.timer import Timer


def cached_basic(path, img, mask, size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/optdisk-%s-%d.bin' % (cache_dir, os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_strength = pickle.load(binary_file)
    else:
        line_strength = basic(img, mask, size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_strength, binary_file)

    binary_file.close()

    return line_strength


def basic(img, mask, size):
    img = img.astype(np.int16)
    bool_mask = mask.astype(np.bool)
    lines, wings = line_factory.generate_lines(size)

    queue = mp.Queue()
    cpu_count = psutil.cpu_count()

    processes = [
        mp.Process(target=basic_worker, args=(img, bool_mask, lines, wings, queue, cpu_count, cpu_id))
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


def basic_worker(img, bool_mask, lines, wings, queue, cpu_count, cpu_id):
    h, w = img.shape[:2]
    slice_height = h // cpu_count
    y_start = cpu_id * slice_height
    linestr = np.zeros((slice_height, w), np.int16)

    for Y in range(y_start, (cpu_id + 1) * slice_height):
        for X in range(w):
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

                    if x < 0 or x >= w or y < 0 or y >= h or not bool_mask[y, x]:
                        continue

                    line_count += 1
                    line_sum += img[y, x]

                if line_count == 0:
                    continue

                line_avg = line_sum / line_count

                if line_avg > max_line_avg:
                    max_line_avg = line_avg
                    max_angle = angle

            wing = wings[max_angle]
            p = Y + wing[0][1], X + wing[0][0]
            q = Y + wing[1][1], X + wing[1][0]
            linestr[Y - y_start, X] = abs(img[p] - img[q])

    queue.put((cpu_id, linestr))


def cache_all():
    timer = Timer()

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        img = 255 - img[:, :, 1]

        for size in range(15, 26, 2):
            timer.start('%s/%d' % (path, size))
            basic(img, mask, size)
            timer.finish()


def main():
    path, img, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(2)
    img = 255 - img[:, :, 1]
    size = 15
    timer = Timer()

    timer.start('Optic disk')
    optic = cached_basic(path, img, mask, size)
    timer.finish()

    optic = normalize_masked(optic, mask)
    th, optic = cv2.threshold(optic, 75, 255, cv2.THRESH_BINARY)
    optic = cv2.erode(optic, np.ones((3, 3), np.uint8), iterations=1)
    img[optic == 255] = 255

    cv2.imshow('Image', img)
    cv2.imshow('Optic disk', optic)
    cv2.imshow('Ground truth', ground_truth)
    cv2.imwrite('C:/Users/Rendicahya/Desktop/optic.jpg', optic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
