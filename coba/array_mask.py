import numpy as np
from util.image_util import normalize

a, b = 3, 3
n = 7
r = 2
y, x = np.ogrid[-a:n - a, -b:n - b]

mask = np.zeros((n, n), np.uint8)
mask[x * x + y * y <= r * r] = 255
image = np.vstack([np.arange(2, 9) for _ in range(7)])
result = image[mask == 255] + 10

print(mask)
print(image)
print(result)

image[mask == 255] = result

print(image)
