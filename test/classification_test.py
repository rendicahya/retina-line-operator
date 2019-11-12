import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

from classification.FeatureExtractor import FeatureExtractor
from dataset.DriveDatasetLoader import DriveDatasetLoader
from util.print_color import *
from util.timer import Timer


def main():
    drive = DriveDatasetLoader('D:/Datasets/DRIVE', 10)
    n_points = 2000
    all_fg_feat = None
    all_bg_feat = None
    size = 15
    timer = Timer()

    timer.start('Feature extraction')

    for path, img, mask, ground in drive.load_train():
        img = 255 - img[:, :, 1]
        feat_extractor = FeatureExtractor(img, mask, path, size, ground, n_points)

        pixel_feat_fg, pixel_feat_bg = feat_extractor.get_pixel_feat()
        single_fg, single_bg = feat_extractor.get_single_feat()
        multi_fg, multi_bg = feat_extractor.get_multi_feat()

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

    timer.stop()

    target = np.append(np.repeat(1, n_points // 2 * 20), np.repeat(0, n_points // 2 * 20))
    classifier = svm.SVC()
    # classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='sqrt', random_state=0,
    #                                     n_jobs=-1)

    timer.start('Training')
    classifier.fit(all_feats, target)
    timer.stop()

    acc_list = []
    acc_fov_list = []

    for path, img, mask, ground in drive.load_test():
        # timer.start(f'Predict {path}')
        img = 255 - img[:, :, 1]
        feat_extractor = FeatureExtractor(img, mask, path, size)

        pixel_feat = feat_extractor.get_pixel_feat()
        single_feat = feat_extractor.get_single_feat()
        multi_feat = feat_extractor.get_multi_feat()

        all_feats = np.column_stack((
            pixel_feat,
            single_feat,
            multi_feat
        ))

        result = classifier.predict(all_feats)

        result[result == 1] = 255
        result_img = np.zeros(mask.shape, np.float64)
        result_img[mask == 255] = result

        acc = accuracy_score(result_img.ravel(), ground.ravel())
        acc_fov = accuracy_score(result, ground[mask == 255].ravel())

        acc_list.append(acc)
        acc_fov_list.append(acc_fov)

        # timer.stop()
        # green(f'Accuracy: {acc}')
        # green(f'Accuracy FOV: {acc_fov}')

        green(acc, ' ')
        green(acc_fov)

        # cv2.imshow(f'Image {path}', img)
        # cv2.imshow(f'Segmentation {path}', result_img)
        # cv2.imshow(f'Ground truth {path}', ground)
        # cv2.imwrite('C:/Users/Randy Cahya Wihandik/Desktop/segmentation.jpg', result_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    yellow('Average accuracy: %f' % np.mean(acc_list))
    yellow('Average FOV accuracy: %f' % np.mean(acc_fov_list))


if __name__ == '__main__':
    main()
