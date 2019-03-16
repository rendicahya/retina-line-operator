import os.path
import pickle

from dataset.DriveDatasetLoader import DriveDatasetLoader
from util.image_util import *
from util.time import Time


def cached_basic(path, image, mask, window_size):
    dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(dir):
        os.mkdir(dir)

    file_path = dir + '/window-%s-%d.bin' % (os.path.basename(path), window_size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        image = pickle.load(binary_file)
    else:
        image = basic(image, mask, window_size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(image, binary_file)

    binary_file.close()

    return image


def basic(image, mask, window_size):
    height, width = image.shape
    half = window_size // 2
    output = np.zeros((height, width), np.float64)
    image = cv2.bitwise_and(image, image, mask=mask)
    bool_mask = mask.astype(np.bool)

    for Y in range(height):
        for X in range(width):
            if not bool_mask[Y, X]:
                continue

            bounds = X - half, X + half + 1, Y - half, Y + half + 1

            image_crop = image[bounds[2]: bounds[3], bounds[0]: bounds[1]]
            mask_crop = mask[bounds[2]: bounds[3], bounds[0]: bounds[1]]

            output[Y, X] = np.sum(image_crop) / np.count_nonzero(mask_crop)

    return output


def cached_integral(path, image, mask, size):
    dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(dir):
        os.mkdir(dir)

    file_path = dir + '/window-%s-%d.bin' % (os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        image = pickle.load(binary_file)
    else:
        image = integral(image, mask, size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(image, binary_file)

    binary_file.close()

    return image


def integral(image, mask, window_size):
    height, width = image.shape[:2]
    half = window_size // 2
    window_avg = np.zeros((height, width), np.float64)

    image = cv2.bitwise_and(image, image, mask=mask)
    image_integral = np.cumsum(np.cumsum(image, 0), 1).astype(np.int32)

    mask[mask > 0] = 1
    bool_mask = mask.astype(np.bool)
    mask_integral = np.cumsum(np.cumsum(mask, 0), 1).astype(np.int32)

    for Y in range(height):
        for X in range(width):
            if not bool_mask[Y, X]:
                continue

            a = Y - half - 1, X - half - 1
            b = Y - half - 1, X + half
            c = Y + half, X - half - 1
            d = Y + half, X + half

            image_sum = image_integral[d] - image_integral[b] - image_integral[c] + image_integral[a]
            mask_sum = mask_integral[d] - mask_integral[b] - mask_integral[c] + mask_integral[a]

            window_avg[Y, X] = image_sum / mask_sum

    return window_avg


def cache_all():
    time = Time()
    size = 15

    for path, image, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        image = 255 - image[:, :, 1]

        time.start('%s/%d' % (path, size))
        cached_integral(path, image, mask, size)
        time.finish()


def main():
    path, image, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing_one(3)
    image = 255 - image[:, :, 1]
    time = Time()
    size = 15

    time.start('Integral')
    window_avg = integral(image, mask, size)
    time.finish()

    cv2.imshow('Image', image)
    cv2.imshow('Window average', normalize_masked(window_avg, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
