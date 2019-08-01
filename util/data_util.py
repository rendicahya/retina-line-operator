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
