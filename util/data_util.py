import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc
from util.numpy_util import to_numpy_array


def normalize(data):
    return cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)


def accuracy(a, b):
    a, b = to_numpy_array(a, b)

    return np.sum(a == b) / a.size


def sensitivity(output, gtruth):
    output, gtruth = to_numpy_array(output, gtruth)

    return np.count_nonzero(output[np.nonzero(gtruth)]) / np.count_nonzero(gtruth)


def specificity(output, gtruth):
    output, gtruth = to_numpy_array(output, gtruth)
    temp = output[gtruth == 0]

    return (temp.size - np.count_nonzero(temp)) / temp.size


def binary_confusion_matrix2(true, pred):
    true = np.copy(true)
    pred = np.copy(pred)

    true[true != 0] = 1
    pred[pred != 0] = 1

    for v in [0, 0], [1, 0], [0, 1], [1, 1]:
        yield np.intersect1d(np.where(pred == v[0]), np.where(true == v[1])).size


def binary_confusion_matrix(true, pred):
    true_classes = np.unique(pred)
    pred_classes = np.unique(pred)

    if not np.array_equal(true_classes, pred_classes):
        print('Unequal classes in pred and true')
        return

    true = replace_ordered(true)
    pred = replace_ordered(pred)

    return np.bincount(true * true_classes.size + pred).reshape((true_classes.size, true_classes.size))


def replace_ordered(arr):
    arr_c = arr.copy()

    for i, v in enumerate(np.unique(arr)):
        arr_c[arr == v] = i

    return arr_c


if __name__ == '__main__':
    arr = np.array([1, 5, 3, 8, 1, 8, 3])

    print(arr)
    arr = replace_ordered(arr)
    print(arr)
