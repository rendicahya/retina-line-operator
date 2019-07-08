import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

from classification.FeatureExtractor import FeatureExtractor
from dataset import DriveDatasetLoader
from util.print_color import blue
from util.time import Time


def proposed():
    drive = DriveDatasetLoader('D:/Datasets/DRIVE', 10)
    N_FEATURES = 1000
    all_fg_feat = None
    all_bg_feat = None
    size = 15
    time = Time()

    time.start('Feature extraction')

    for path, img, mask, ground_truth in drive.load_training():
        img = 255 - img[:, :, 1]
        feat_extractor = FeatureExtractor(img, mask, path, size, ground_truth, N_FEATURES)

        pixel_feat_fg, pixel_feat_bg = feat_extractor.get_pixel_feat()
        single_fg, single_bg = feat_extractor.get_single_linestr_feat()
        multi_fg, multi_bg = feat_extractor.get_multi_linestr_feat()

        fg_feat = np.column_stack((
            pixel_feat_fg,
            single_fg,
            multi_fg
        ))

        bg_feat = np.column_stack((
            pixel_feat_bg,
            single_bg,
            multi_bg
        ))

        all_fg_feat = fg_feat if all_fg_feat is None else np.vstack((all_fg_feat, fg_feat))
        all_bg_feat = bg_feat if all_bg_feat is None else np.vstack((all_bg_feat, bg_feat))

    all_feats = np.vstack((all_fg_feat, all_bg_feat))
    time.finish()

    target = np.append(np.repeat(1, N_FEATURES * 20), np.repeat(0, N_FEATURES * 20))
    classifier = svm.SVC()
    # classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='sqrt', random_state=0,
    #                                     n_jobs=-1)

    time.start('Training')
    classifier.fit(all_feats, target)
    time.finish()
    time.start('Predict')

    for path, img, mask, ground_truth in drive.load_testing():
        img = 255 - img[:, :, 1]
        feat_extractor = FeatureExtractor(img, mask, path, size)

        pixel_feat = feat_extractor.get_pixel_feat()
        single_feat = feat_extractor.get_single_linestr_feat()
        multi_feat = feat_extractor.get_multi_linestr_feat()

        all_feats = np.column_stack((
            pixel_feat,
            single_feat,
            multi_feat
        ))

        result = classifier.predict(all_feats)

        result[result == 1] = 255
        result_image = np.zeros(mask.shape, np.float64)
        result_image[mask == 255] = result

        blue('Accuracy: %f' % accuracy_score(result_image.ravel(), ground_truth.ravel()))
        blue('Accuracy FOV: %f' % accuracy_score(result, ground_truth[mask == 255].ravel()))

        # cv2.imshow('Image', img)
        cv2.imshow('Segmentation', result_image)
        cv2.imshow('Ground truth', ground_truth)
        # cv2.imwrite('C:/Users/Randy Cahya Wihandik/Desktop/segmentation.jpg', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    time.finish()
