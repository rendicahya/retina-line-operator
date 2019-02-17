import numpy as np

a = np.zeros((10, 10), np.uint8)
b = np.zeros((10, 10), np.uint8)
c = np.zeros((10, 10), np.uint8)
c = [a, b, c]
d = np.stack([i for i in c])
e = np.average(d, axis=0)

print(d.shape)
print(e.shape)
