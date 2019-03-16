import os.path
import pickle

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods import window_average, single_line
from util.image_util import *
from util.time import Time


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


def multi(path, image, mask, line_size):
    window_avg = window_average.cached_integral(path, image, mask, line_size)
    line_str = [single_line.single(single_line.cached_line(path, image, mask, size), window_avg, mask) for size in
                range(1, line_size + 1, 2)]

    return np.average(np.stack(line_str), axis=0)


def cache_all():
    time = Time()

    for path, image, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training():
        image = 255 - image[:, :, 1]

        time.start('%s' % path)
        cached_multi(path, image, mask, 15)
        time.finish()


def main():
    path, img, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    img = 255 - img[:, :, 1]
    size = 15
    time = Time()

    time.start('Multi scale')
    multi_scale = cached_multi(path, img, mask, size)
    time.finish()

    # time.start('Find best multi scale')
    # best_multi_thresh, best_multi = find_best_threshold(multi_scale, mask, ground_truth)
    # time.finish()

    # green('Best single scale threshold: %d' % best_single_thresh)
    # green('Best multi scale threshold: %d' % best_multi_thresh)

    cv2.imshow('Image', img)
    cv2.imshow('Multi', normalize_masked(multi_scale, mask))
    # cv2.imshow('Best multi', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Multi histeq', cv2.equalizeHist(multi_scale))
    # cv2.imshow('Ground truth', ground_truth)
    # cv2.imshow('Best binary', binary)
    # cv2.imshow('Multi', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
