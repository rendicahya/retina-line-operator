import cv2
import numpy as np
from skimage import io


class DriveDatasetLoader:

    def __init__(self, dir, erode_mask_by=None):
        self.dir = dir
        self.erode_mask = erode_mask_by

    def load(self, img_path, mask_path, ground_path):
        mask = io.imread(mask_path)
        image = cv2.imread(img_path)
        ground = io.imread(ground_path)

        if self.erode_mask is not None:
            mask = cv2.erode(mask, np.ones((self.erode_mask, self.erode_mask), np.uint8), iterations=1)

        corrected_ground = cv2.bitwise_and(ground, ground, mask=mask)

        return img_path, image, mask, corrected_ground

    def load_training_one(self, data_id):
        data_id += 20
        img_path = '%s/training/images/%02d_training.tif' % (self.dir, data_id)
        mask_path = '%s/training/mask/%02d_training_mask.gif' % (self.dir, data_id)
        ground_path = '%s/training/1st_manual/%02d_manual1.gif' % (self.dir, data_id)

        return self.load(img_path, mask_path, ground_path)

    def load_training(self):
        data = []

        for data_id in range(1, 21):
            data.append(self.load_training_one(data_id))

        return data

    def load_testing_one(self, data_id):
        img_path = '%s/test/images/%02d_test.tif' % (self.dir, data_id)
        mask_path = '%s/test/mask/%02d_test_mask.gif' % (self.dir, data_id)
        ground_path = '%s/test/1st_manual/%02d_manual1.gif' % (self.dir, data_id)

        return self.load(img_path, mask_path, ground_path)

    def load_testing(self):
        data = []

        for data_id in range(1, 21):
            data.append(self.load_testing_one(data_id))

        return data
