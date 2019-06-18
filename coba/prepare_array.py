import numpy as np

a = np.vstack([np.arange(2, 11) for _ in range(5)])

print(a)
print(a.ravel())
print(a.ravel().reshape(5, -1))
