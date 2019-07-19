import cv2

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.image_util import normalize_masked, subtract_masked
from util.timer import Timer
from util.image_util import find_best_thresh


def cached_single(path, img, mask, size):
    window_avg = cached_integral(path, img, mask, size)
    line_img = cached_statistics(path, img, mask, size)['max']

    return subtract_masked(line_img, window_avg, mask)


def cached_single_norm(path, img, mask, size):
    line_str = cached_single(path, img, mask, size)

    return normalize_masked(line_str, mask)


def main():
    path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)
    img = 255 - img[:, :, 1]
    size = 15
    timer = Timer()

    timer.start('Single')
    line_str = cached_single_norm(path, img, mask, size)
    timer.stop()

    # bin = cv2.threshold(line_str, 65, 255, cv2.THRESH_BINARY)[1]

    # timer.start('Single scale + wing')
    # single_scale_wing = single(img, mask, window_avg, size)
    # timer.finish()

    timer.start('Find best threshold')
    thresh, single_thresh, acc = find_best_thresh(line_str, ground, mask)
    timer.stop()

    print(thresh)

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
    cv2.imshow('Single', 255 - line_str)
    cv2.imshow('Single thresh', 255 - single_thresh)
    # cv2.imshow('Single + wing', normalize_masked(255 - single_scale_wing, mask))
    # cv2.imshow('Single best', 255 - normalize_masked(best_single, mask))
    # cv2.imshow('Multi', normalize_masked(multi_scale, mask))
    # cv2.imshow('Best multi', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Multi histeq', cv2.equalizeHist(multi_scale))
    # cv2.imshow('Ground truth', ground)
    # cv2.imshow('Binary', bin)
    # cv2.imshow('Multi', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single.png', 255 - line_str)
    cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single-thresh.png', 255 - single_thresh)


if __name__ == '__main__':
    main()
