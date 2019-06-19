from dataset import DriveDatasetLoader
from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.single_line_opr import cached_line, subtract
from methods.window_average import cached_integral
from util.time import Time


def main():
    drive = DriveDatasetLoader('D:/Datasets/DRIVE', 10)
    size = 15

    for path, img, mask, ground_truth in drive.load_training():
        img = 255 - img[:, :, 1]
        time = Time()

        time.start(path)
        window_avg = cached_integral(path, img, mask, size)
        line_img = cached_line(path, img, mask, size)
        single_img = subtract(line_img, window_avg, mask)
        time.finish()


if __name__ == '__main__':
    main()
