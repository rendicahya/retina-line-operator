import numpy as np

a = np.zeros((3, 3), np.uint8)
b = np.ones((3, 3), np.uint8)

c = a[0, 0] - b[0, 0] - 1

# print(a)
# print(b)
print(c)
