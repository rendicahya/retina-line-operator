import numpy as np

a = np.zeros((10, 10), np.uint8)
b = np.zeros((10, 10), np.uint8)

b[5, 5] = 1

print(a)
print(b)
# print(np.subtract(a, b))
print(np.argwhere(np.subtract(a, b)))
