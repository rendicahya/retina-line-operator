import cv2

from dataset.DriveDatasetLoader import DriveDatasetLoader
from util.image_util import clahe

drive = DriveDatasetLoader('D:/Datasets/DRIVE', 10)
img = drive.load_training_one(2)[1]
# img = 255 - img[:, :, 1]

# cv2.imshow('Image', img)
# cv2.imshow('CLAHE', clahe(img))
cv2.imwrite(r'D:\Google Drive\Penelitian\IJIGSP 2019\rgb.jpg', img)
cv2.imwrite(r'D:\Google Drive\Penelitian\IJIGSP 2019\r.jpg', img[:, :, 2])
cv2.imwrite(r'D:\Google Drive\Penelitian\IJIGSP 2019\g.jpg', img[:, :, 1])
cv2.imwrite(r'D:\Google Drive\Penelitian\IJIGSP 2019\b.jpg', img[:, :, 0])
cv2.imwrite(r'D:\Google Drive\Penelitian\IJIGSP 2019\clahe.jpg', clahe(img[:, :, 1]))
cv2.waitKey(0)
