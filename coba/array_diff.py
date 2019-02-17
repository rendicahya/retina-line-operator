import numpy as np

a = np.array([
    [255, 255, 0, 0, 0],
    [255, 0, 0, 255, 0],
    [0, 0, 255, 0, 255],
    [255, 0, 255, 0, 0],
    [0, 255, 0, 0, 255]
], np.uint8)

b = np.array([
    [0, 0, 255, 255, 0],
    [0, 255, 255, 0, 0],
    [255, 0, 255, 0, 0],
    [255, 255, 0, 0, 0],
    [255, 255, 0, 0, 0]
], np.uint8)

print(np.sum(a == b) / a.size)
