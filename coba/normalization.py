import cv2
import numpy as np

a = np.vstack([np.arange(2, 11) for _ in range(5)])
norm = cv2.normalize(a, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)

print(a)
print(norm)

print(a.shape)
print(norm.shape)
