import numpy as np


def to_numpy_array(*args):
    if len(args) == 1:
        return args[0] if type(args[0]) is np.ndarray else np.array(args[0])

    return [item if type(item) is np.ndarray else np.array(item) for item in args]
