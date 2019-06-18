import numpy as np
from sklearn.metrics import classification_report

a = np.zeros((4, 4), np.uint8)
b = a.copy()

a[2:, 2:] = 1
b[1:-1, 1:-1] = 1

print(a)
print(b)

print(classification_report(b.ravel(), a.ravel()))
