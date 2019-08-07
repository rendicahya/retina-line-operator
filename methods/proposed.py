import cv2

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.image_util import subtract_line_str, gray_norm


def proposed_norm(path, img, mask, size):
    window = cached_integral(path, img, mask, size)
    stat = cached_statistics(path, img, mask, size)
    sub = subtract_line_str(stat['min'], window, mask)
    min_window = gray_norm(sub, mask)
    min_window[mask == 255] = 255 - min_window[mask == 255]

    return min_window


def main():
    path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)
    img = 255 - img[:, :, 1]
    size = 15

    proposed = proposed_norm(path, img, mask, size)
    bin = cv2.threshold(proposed, 127, 255, cv2.THRESH_BINARY)[1]
    img[bin == 255] = 255

    cv2.imshow('Image', img)
    cv2.imshow('Proposed', proposed)
    cv2.imshow('Bin', bin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
