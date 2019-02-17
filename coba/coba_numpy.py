import numpy as np

N_FEATURES = 3

target = np.append([1] * N_FEATURES * 20, [0] * N_FEATURES * 20)
target2 = np.append(np.repeat(1, N_FEATURES * 20), np.repeat(0, N_FEATURES * 20))

print(target)
print(target2)
