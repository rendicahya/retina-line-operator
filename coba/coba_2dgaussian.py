import matplotlib.pylab as plt
import numpy as np
from numpy import pi, exp, sqrt

s, k = 3, 7
probs = [exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)]
kernel = np.outer(probs, probs)
kernel *= (1 / np.max(kernel))

print(kernel)
print(np.max(kernel))
print(np.min(kernel))

plt.imshow(kernel)
plt.colorbar()
plt.show()
