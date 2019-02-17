import numpy as np
from util.data_util import *

a = np.array([
    [255, 255, 0],
    [255, 0, 255],
    [0, 0, 255]
])

b = np.array([
    [255, 255, 255],
    [255, 255, 0],
    [0, 0, 0]
])

print(accuracy(a, b))
