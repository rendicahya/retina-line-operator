import multiprocessing as mp
import os.path
import pickle

import cv2
import numpy as np
import psutil

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.line_factory import generate_lines
from methods.window_average import cached_integral
from util.image_util import subtract_line_str, gray_norm, find_best_thresh
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

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_train():
        img = 255 - img[:, :, 1]

        time.start(path)
        cached_statistics(path, img, mask, size)
        time.stop()


def main():
    path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(8)

    img = 255 - img[:, :, 1]
    size = 15

    window = cached_integral(path, img, mask, size)
    stat = cached_statistics(path, img, mask, size)
    single = subtract_line_str(stat['max'], window, mask)
    single = gray_norm(single, mask)
    thresh, single, _ = find_best_thresh(single, ground, mask)
    single = 255 - cv2.threshold(single, thresh, 255, cv2.THRESH_BINARY)[1]
    # single[mask == 0] = 0
    # img[single == 255] = 255
    # img[mask == 0] = 255

    # print(np.min(single[mask == 255]))
    # print(np.max(single[mask == 255]))

    # cv2.imshow('Image', img)
    # cv2.imshow('Single', single[150:220, 418:491])
    # cv2.imshow('Single', single)
    # cv2.imshow('Ground', ground)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single-false-bin.jpg', single[150:220, 418:491])


def validation():
    path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    img = 255 - img[:, :, 1]
    size = 15

    window = cached_integral(path, img, mask, size)
    stat = cached_statistics(path, img, mask, size)
    mi = stat['min']
    a = 189, 149
    b = 179, 265
    c = 131, 170

    print('Condition a')
    print(mi[a])
    print(window[a])

    print('Condition b')
    print(mi[b])
    print(window[b])

    print('Condition c')
    print(mi[c])
    print(window[c])


if __name__ == '__main__':
    main()
