from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.multi_line_opr import cached_multi_norm, cached_multi
from methods.single_line_opr import cached_single
from util.print_color import *
from util.test.basic_test_util import basic_train, basic_get_acc, basic_calc_auc, basic_test_each
from util.test.optic_test_util import optic_train, optic_test_each, optic_get_acc
from util.test.proposed_test_util import proposed_train, proposed_get_acc
from util.timer import Timer


def basic_training():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    # op = cached_single_norm
    op = cached_multi_norm

    size = 15
    timer = Timer()

    green('basic_training')
    timer.start('Train')
    thresh, train_acc = basic_train(op, train_data, size)
    timer.stop()

    timer.start('Test')
    test_acc = basic_get_acc(op, test_data, thresh, size)
    timer.stop()

    green(f'Threshold: {thresh}')
    green(f'Train average accuracy: {train_acc}')
    green(f'Test average accuracy: {test_acc}')

    # cv2.imshow('Image', linestr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def basic_each():
    # data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    # op = cached_single_norm
    op = cached_multi_norm

    size = 15
    timer = Timer()

    timer.start('basic_each')
    acc = basic_test_each(op, data, size)
    auc = basic_calc_auc(data, op, size)
    timer.stop()

    green(f'Accuracy: {acc}')
    green(f'AUC: {auc}')


def optic_training():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    # op = cached_single
    # thresh = 64

    op = cached_multi
    thresh = 119

    size = 15
    timer = Timer()

    green('optic_training')
    timer.start('Train')
    disk_thresh, train_acc = optic_train(op, thresh, train_data, size)
    timer.stop()

    timer.start('Test')
    test_acc = optic_get_acc(op, test_data, thresh, disk_thresh)
    timer.stop()

    blue(f'Disk threshold: {disk_thresh}')
    blue(f'Train average accuracy: {train_acc}')
    blue(f'Test average accuracy: {test_acc}')


def optic_each():
    # data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    # op = cached_single
    op = cached_multi

    size = 15
    timer = Timer()

    timer.start('optic_no_training')
    acc, auc = optic_test_each(op, data, size)
    timer.stop()

    green(f'Test average accuracy: {acc}')
    green(f'Test average AUC: {auc}')


def proposed_training():
    train_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    test_data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    op = cached_single
    # op = cached_multi

    thresh = 64
    optic_thresh = 42
    size = 15
    timer = Timer()

    green('proposed_with_training')
    timer.start('Train')
    proposed_thresh, train_acc = proposed_train(op, thresh, optic_thresh, train_data, size)
    timer.stop()

    timer.start('Test')
    test_acc = proposed_get_acc(op, test_data, thresh, optic_thresh, proposed_thresh)
    timer.stop()

    blue(f'Proposed threshold: {proposed_thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')


def proposed_each():
    pass


if __name__ == '__main__':
    basic_each()
