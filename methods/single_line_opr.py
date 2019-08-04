from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.data_util import auc_score
from util.image_util import find_best_thresh
from util.image_util import norm_masked, subtract_masked
from util.timer import Timer


def cached_single(path, img, mask, size):
    window_avg = cached_integral(path, img, mask, size)
    line_img = cached_statistics(path, img, mask, size)['max']

    return subtract_masked(line_img, window_avg, mask)


def cached_single_norm(path, img, mask, size):
    line_str = cached_single(path, img, mask, size)

    return norm_masked(line_str, mask)


def main():
    path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)
    img = 255 - img[:, :, 1]
    size = 15
    timer = Timer()

    timer.start('Single')
    line_str = cached_single(path, img, mask, size)
    timer.stop()

    timer.start('AUC')
    auc = auc_score(ground, line_str, mask)
    timer.stop()

    print(ground.min())
    print(ground.max())
    print(line_str.min())
    print(line_str.max())

    timer.start('Find best threshold')
    thresh, single_thresh, acc = find_best_thresh(line_str, ground, mask)
    timer.stop()

    print(auc)
    print(thresh)

    # cv2.imshow('Image', img)
    # cv2.imshow('Single', line_str)
    # cv2.imshow('Single thresh', single_thresh)
    # cv2.imshow('Ground truth', ground)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single.png', 255 - line_str)
    # cv2.imwrite(r'C:\Users\Randy Cahya Wihandik\Desktop\single-thresh.png', 255 - single_thresh)


if __name__ == '__main__':
    main()
