import cv2

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.image_util import find_best_thresh
from util.image_util import normalize_masked, subtract_masked
from util.timer import Timer


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

    timer.start('Find best threshold')
    thresh, single_thresh, acc, fpr_list, tpr_list = find_best_thresh(line_str, ground, mask)
    timer.stop()

    print(thresh)

    cv2.imshow('Image', img)
    cv2.imshow('Single', line_str)
    cv2.imshow('Single thresh', single_thresh)
    cv2.imshow('Ground truth', ground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single.png', 255 - line_str)
    # cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single-thresh.png', 255 - single_thresh)


if __name__ == '__main__':
    main()
