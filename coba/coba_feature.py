from classification.FeatureExtractor import FeatureExtractor
from dataset.DriveDatasetLoader import DriveDatasetLoader

path, image, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE').load_training_one(1)
image = 255 - image[:, :, 1]
feat_ex = FeatureExtractor(image, mask, path, 15)

print(feat_ex.get_pixel_feat().shape)
