import cv2
import numpy as np

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.single_line_opr import cached_single_norm
from methods.multi_line_opr import cached_multi_norm
from util.data_util import accuracy
from util.time import Time


def test(op, data):
    best_acc = 0
    best_thresh = 0
    size = 15
    time = Time()

    for thresh in range(1, 2):
        acc_list = []

        time.start(thresh)

        for path, img, mask, ground in data:
            img = 255 - img[:, :, 1]

            line_str = op(path, img, mask, size)
            cv2.imshow(path, line_str)
            bin = cv2.threshold(line_str, thresh, 255, cv2.THRESH_BINARY)[1]
            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        avg = np.average(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

        time.finish()
        print(f'{avg}')

    print(f'Best threshold: {best_thresh}')
    print(f'Best accuracy: {best_acc}')


if __name__ == '__main__':
    data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    test(cached_multi_norm, data)

    # cv2.imshow('Image', linestr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
