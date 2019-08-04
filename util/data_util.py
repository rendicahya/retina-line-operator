import cv2
import numpy as np

from util.numpy_util import to_numpy_array


def normalize(data):
    return cv2.normalize(data, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)


def accuracy(true, pred):
    true, pred = to_numpy_array(true, pred)

    return np.sum(true == pred) / true.size


def sensitivity(pred, true):
    pred, true = to_numpy_array(pred, true)

    return np.count_nonzero(pred[np.nonzero(true)]) / np.count_nonzero(true)


def specificity(pred, true):
    pred, true = to_numpy_array(pred, true)
    temp = pred[true == 0]

    return (temp.size - np.count_nonzero(temp)) / temp.size


def confusion_matrix(true, pred):
    true_classes = np.unique(pred)
    pred_classes = np.unique(pred)

    if not np.array_equal(true_classes, pred_classes):
        print('Unequal classes in pred and true')
        return None

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
    print(replace_ordered(arr))
