import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

from classification.FeatureExtractor import FeatureExtractor
from dataset import DriveDatasetLoader
from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.multi_line_opr import cached_multi
from methods.single_line_opr import cached_line, subtract
from methods.window_average import cached_integral
from util.data_util import accuracy
from util.image_util import normalize_masked
from util.print_color import *
from util.time import Time


def test(op):
    best_acc = 0
    best_thresh = 0
    size = 15

    for thresh in range(1, 255):
        acc_list = []

        for path, img, mask, ground in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
            img = 255 - img[:, :, 1]
            time = Time()

            time.start(path)
            linestr = op(path, img, mask, size)
            bin = cv2.threshold(linestr, thresh, 255, cv2.THRESH_BINARY)[1]
            bin_fov = bin[mask == 255]
            ground_fov = ground[mask == 255]
            acc = accuracy(bin_fov, ground_fov)

            acc_list.append(acc)
            time.finish()

        avg = np.average(acc_list)

        if avg > best_acc:
            best_acc = avg
            best_thresh = thresh

        print(f'{thresh} {avg}')

    print(f'Best threshold: {best_thresh}')
    print(f'Best accuracy: {best_acc}')


def single(path, img, mask, size):
    window_avg = cached_integral(path, img, mask, size)
    line_img = cached_line(path, img, mask, size)
    linestr = subtract(line_img, window_avg, mask)
    linestr = normalize_masked(linestr, mask)

    return linestr


def multi(path, img, mask, size):
    linestr = cached_multi(path, img, mask, size)
    linestr = normalize_masked(linestr, mask)

    return linestr


def classification():
    drive = DriveDatasetLoader('D:/Datasets/DRIVE', 10)
    N_FEATURES = 1000
    all_fg_feat = None
    all_bg_feat = None
    line_size = 15
    time = Time()

    time.start('Feature extraction')

    for path, img, mask, ground_truth in drive.load_training():
        img = 255 - img[:, :, 1]
        feat_extractor = FeatureExtractor(img, mask, path, line_size, ground_truth, N_FEATURES)

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

    all_feat = np.vstack((all_fg_feat, all_bg_feat))
    time.finish()

    target = np.append(np.repeat(1, N_FEATURES * 20), np.repeat(0, N_FEATURES * 20))
    classifier = svm.SVC()
    # classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', max_features='sqrt', random_state=0,
    #                                     n_jobs=-1)

    time.start('Training')
    classifier.fit(all_feat, target)
    time.finish()

    path, img, mask, ground_truth = drive.load_testing_one(1)
    img = 255 - img[:, :, 1]
    feat_extractor = FeatureExtractor(img, mask, path, line_size)

    pixel_feat = feat_extractor.get_pixel_feat()
    single_feat = feat_extractor.get_single_linestr_feat()
    multi_feat = feat_extractor.get_multi_linestr_feat()

    all_feat = np.column_stack((
        pixel_feat,
        single_feat,
        multi_feat
    ))

    time.start('Predict')
    result = classifier.predict(all_feat)
    time.finish()

    result[result == 1] = 255
    result_image = np.zeros(mask.shape, np.float64)
    result_image[mask == 255] = result

    blue('Accuracy: %f' % accuracy_score(result_image.ravel(), ground_truth.ravel()))
    blue('Accuracy FOV: %f' % accuracy_score(result, ground_truth[mask == 255].ravel()))

    # cv2.imshow('Image', img)
    cv2.imshow('Segmentation', result_image)
    cv2.imshow('Ground truth', ground_truth)
    cv2.imwrite('C:/Users/Randy Cahya Wihandik/Desktop/segmentation.jpg', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # test(single)
