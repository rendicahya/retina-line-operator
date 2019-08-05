from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.multi_line_opr import cached_multi_norm
from methods.single_line_opr import cached_single_norm
from util.print_color import *
from util.test_util import find_best_acc, get_accuracy, find_best_acc_disk, find_best_acc_proposed, get_accuracy_disk, \
    get_accuracy_proposed
from util.timer import Timer


def test_line():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()
    op = cached_single_norm
    # op = cached_multi_norm
    timer = Timer()

    timer.start('Train')
    thresh, train_acc = find_best_acc(op, train_data)
    timer.stop()

    timer.start('Test')
    test_acc = get_accuracy(op, test_data, thresh)
    timer.stop()

    blue(f'Threshold: {thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')

    # cv2.imshow('Image', linestr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def test_optic():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()
    # op = cached_single_norm
    op = cached_multi_norm
    thresh = 119
    timer = Timer()

    timer.start('Train')
    disk_thresh, train_acc = find_best_acc_disk(op, thresh, train_data)
    timer.stop()

    timer.start('Test')
    test_acc = get_accuracy_disk(op, test_data, thresh, disk_thresh)
    timer.stop()

    blue(f'Disk threshold: {disk_thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')


def test_proposed():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()
    op = cached_single_norm
    # op = cached_multi_norm
    thresh = 64
    timer = Timer()

    timer.start('Train')
    proposed_thresh, train_acc = find_best_acc_proposed(op, thresh, train_data)
    timer.stop()

    timer.start('Test')
    test_acc = get_accuracy_proposed(op, test_data, thresh, proposed_thresh)
    timer.stop()

    blue(f'Proposed threshold: {proposed_thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')


if __name__ == '__main__':
    test_proposed()
