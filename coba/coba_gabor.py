import cv2
import numpy as np

data_id = 21
image = 255 - cv2.imread('D:/Datasets/DRIVE/training/images/%d_training.tif' % data_id)[:, :, 1]

kernel = cv2.getGaborKernel((3, 3), 3, np.pi / 4, 5, .5)
filtered = cv2.filter2D(image, -1, kernel)

cv2.imshow('Gabor', filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
