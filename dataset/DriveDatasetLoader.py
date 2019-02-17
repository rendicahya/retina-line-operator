import cv2
import numpy as np
from skimage import io


class DriveDatasetLoader:

    def __init__(self, dir, erode_mask_by=None):
        self.dir = dir
        self.erode_mask = erode_mask_by

    def load(self, image_path, mask_path, ground_truth_path):
        mask = io.imread(mask_path)
        image = cv2.imread(image_path)
        ground_truth = io.imread(ground_truth_path)

        if self.erode_mask is not None:
            mask = cv2.erode(mask, np.ones((self.erode_mask, self.erode_mask), np.uint8), iterations=1)

        corrected_ground_truth = cv2.bitwise_and(ground_truth, ground_truth, mask=mask)

        return image_path, image, mask, corrected_ground_truth

    def load_training_one(self, data_id):
        data_id += 20
        image_path = '%s/training/images/%02d_training.tif' % (self.dir, data_id)
        mask_path = '%s/training/mask/%02d_training_mask.gif' % (self.dir, data_id)
        ground_truth_path = '%s/training/1st_manual/%02d_manual1.gif' % (self.dir, data_id)

        return self.load(image_path, mask_path, ground_truth_path)

    def load_training(self):
        for data_id in range(1, 21):
            yield self.load_training_one(data_id)

    def load_testing_one(self, data_id):
        image_path = '%s/test/images/%02d_test.tif' % (self.dir, data_id)
        mask_path = '%s/test/mask/%02d_test_mask.gif' % (self.dir, data_id)
        ground_truth_path = '%s/test/1st_manual/%02d_manual1.gif' % (self.dir, data_id)

        return self.load(image_path, mask_path, ground_truth_path)

    def load_testing(self):
        for data_id in range(1, 21):
            yield self.load_testing_one(data_id)
