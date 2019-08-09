from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.multi_line_opr import cached_multi_norm, cached_multi
from methods.single_line_opr import cached_single, cached_single_norm
from util.basic_test_util import basic_train, basic_get_acc, calc_auc, basic_test_each
from util.print_color import *
from util.test.optic_test_util import optic_train, optic_test_each, optic_get_acc
from util.test.proposed_test_util import train_proposed, get_accuracy_proposed
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
    train_auc = calc_auc(train_data, op, size)
    timer.stop()

    timer.start('Test')
    test_acc = basic_get_acc(op, test_data, thresh, size)
    test_auc = calc_auc(test_data, op, size)
    timer.stop()

    green(f'Threshold: {thresh}')
    green(f'Train average accuracy: {train_acc}')
    green(f'Train average AUC: {train_auc}')
    green(f'Test average accuracy: {test_acc}')
    green(f'Test average AUC: {test_auc}')

    # cv2.imshow('Image', linestr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def basic_each():
    data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training()
    # data = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing()

    op = cached_single_norm
    # op = cached_multi_norm

    size = 15
    timer = Timer()

    timer.start('line_no_training')
    acc = basic_test_each(op, data, size)
    timer.stop()

    green(f'Test average accuracy: {acc}')


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
    proposed_thresh, train_acc = train_proposed(op, thresh, optic_thresh, train_data, size)
    timer.stop()

    timer.start('Test')
    test_acc = get_accuracy_proposed(op, test_data, thresh, optic_thresh, proposed_thresh)
    timer.stop()

    blue(f'Proposed threshold: {proposed_thresh}')
    blue(f'Train accuracy: {train_acc}')
    blue(f'Test accuracy: {test_acc}')


def proposed_each():
    pass


if __name__ == '__main__':
    proposed_training()
