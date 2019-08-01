import numpy as np

from dataset.DriveDatasetLoader import DriveDatasetLoader

np.random.seed(0)

path, img, mask, ground = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)
all_fg_idx = np.argwhere(ground == 255)
rand_fg_idx = np.random.choice(all_fg_idx.shape[0], 10, False)
selected_fg_idx = all_fg_idx[rand_fg_idx]

# print(all_fg_idx)
# print(len(all_fg_idx))
print(rand_fg_idx)
# print(len(rand_fg_idx))
# print(selected_fg_idx)
# print(len(selected_fg_idx))

np.random.seed(1)

rand_fg_idx = np.random.choice(all_fg_idx.shape[0], 10, False)
print(rand_fg_idx)
