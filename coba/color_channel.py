from dataset.DriveDatasetLoader import DriveDatasetLoader
from util.image_util import *

path, img, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)
b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

cv2.imwrite('C:/Users/Rendicahya/Desktop/b.jpg', b)
cv2.imwrite('C:/Users/Rendicahya/Desktop/g.jpg', g)
cv2.imwrite('C:/Users/Rendicahya/Desktop/r.jpg', r)
