import cv2
from dataset.DriveDatasetLoader import DriveDatasetLoader
import numpy as np

drive = DriveDatasetLoader('D:/Datasets/DRIVE', 10)
img = drive.load_training_one(1)[2]

print(np.unique(img))
