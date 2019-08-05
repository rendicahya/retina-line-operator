import cv2

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.data_util import auc_score
from util.image_util import find_best_thresh
from util.image_util import gray_norm, subtract_masked


def cached_single(path, img, mask, size):
    window_avg = cached_integral(path, img, mask, size)
    line_img = cached_statistics(path, img, mask, size)['max']

    return subtract_masked(line_img, window_avg, mask)


def cached_single_norm(path, img, mask, size):
    line_str = cached_single(path, img, mask, size)

    return gray_norm(line_str, mask)


def main():
    path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)
    img = 255 - img[:, :, 1]
    size = 15
    line_str = cached_single(path, img, mask, size)
    auc = auc_score(ground, line_str, mask)
    thresh, bin, acc = find_best_thresh(line_str, ground, mask)

    print('AUC:', auc)
    print('Acc:', acc)
    print('Thresh:', thresh)

    cv2.imshow('Image', img)
    cv2.imshow('Single', gray_norm(line_str, mask))
    cv2.imshow('Binary', bin)
    cv2.imshow('Ground truth', ground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single.png', 255 - line_str)
    # cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single-thresh.png', 255 - bin)


if __name__ == '__main__':
    main()
