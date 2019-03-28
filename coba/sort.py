import numpy as np

a = np.arange(24).reshape((2, 3, 4))

print(a.shape)
print(a)

# a = np.sort(a)
a = a[::-1]

print(a)
