from dataset import DriveDatasetLoader
from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.single_line_opr import cached_line, subtract
from methods.window_average import cached_integral
from util.image_util import find_thresh_all
from util.image_util import find_thresh_one
from util.image_util import normalize_masked
from util.print_color import *
from util.time import Time


def find_best_each():
    for img_id in range(1, 5):
        path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing_one(img_id)
        img = 255 - img[:, :, 1]
        size = 15
        time = Time()

        time.start(path)
        window_avg = cached_integral(path, img, mask, size)
        line_img = cached_line(path, img, mask, size)
        single_img = subtract(line_img, window_avg, mask)
        single_img = normalize_masked(single_img, mask)
        best_thresh, best_img, best_acc = find_thresh_one(single_img, ground, mask)
        time.finish()

        green(f'Best threshold: {best_thresh}')

        # cv2.imshow('Single Line Operator', single_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


def main():
    best_thresh, best_acc = find_thresh_all(DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing())

    print(best_acc)


if __name__ == '__main__':
    main()
