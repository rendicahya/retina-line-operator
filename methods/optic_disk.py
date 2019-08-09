import multiprocessing as mp
import os.path
import pickle
import sys

import cv2
import numpy as np
import psutil

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods import line_factory
from util.image_util import gray_norm
from util.timer import Timer


def cached_optic_norm(path, img, mask, size):
    optic = cached_optic(path, img, mask, size)

    return gray_norm(optic, mask)


def cached_optic(path, img, mask, size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/disk-%s-%d.bin' % (cache_dir, os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_strength = pickle.load(binary_file)
    else:
        line_strength = optic(img, mask, size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_strength, binary_file)

    binary_file.close()

    return line_strength


def optic(img, mask, size):
    img = img.astype(np.int16)
    bool_mask = mask.astype(np.bool)
    lines, wings = line_factory.generate_lines(size)

    queue = mp.Queue()
    cpu_count = psutil.cpu_count()

    processes = [
        mp.Process(target=optic_worker, args=(img, bool_mask, lines, wings, queue, cpu_count, cpu_id))
        for cpu_id in range(cpu_count)]

    for p in processes:
        p.start()

    slices = [queue.get() for _ in processes]
    slices = sorted(slices, key=lambda s: s[0])
    slices = [piece[1] for piece in slices]

    for p in processes:
        p.join()

    result = np.vstack(slices)
    mask = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    result[mask == 0] = 0

    return result


def optic_worker(img, bool_mask, lines, wings, queue, cpu_count, cpu_id):
    h, w = img.shape[:2]
    slice_height = h // cpu_count
    y_start = cpu_id * slice_height
    line_str = np.zeros((slice_height, w), np.int16)

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
            line_str[Y - y_start, X] = abs(img[p] - img[q])

    queue.put((cpu_id, line_str))


def save_cache():
    timer = Timer()
    size = 15

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        img = 255 - img[:, :, 1]

        timer.start(path)
        cached_optic(path, img, mask, size)
        timer.stop()


def main():
    pass
    # path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(5)
    # img = 255 - img[:, :, 1]
    # size = 15
    #
    # disk = cached_disk_norm(path, img, mask, size)
    # disk = cv2.threshold(disk, 75, 255, cv2.THRESH_BINARY)[1]
    # disk = cv2.erode(disk, np.ones((3, 3), np.uint8), iterations=1)
    #
    # line_str = cached_multi(path, img, mask, size)
    # line_str[disk == 255] = line_str[mask == 255].min()
    # line_str = gray_norm(line_str, mask)
    # bin = find_best_thresh(line_str, ground, mask)[1]
    #
    # cv2.imshow('Image', img)
    # cv2.imshow('Line', line_str)
    # cv2.imshow('Binary', bin)
    # cv2.imshow('Optic disk', disk)
    # cv2.imshow('Ground truth', ground)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\optic.png', img)


if __name__ == '__main__':
    main()
