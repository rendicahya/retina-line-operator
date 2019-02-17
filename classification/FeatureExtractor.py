import cv2
import numpy as np

from methods import window_average, line_operator
from util.data_util import normalize


class FeatureExtractor:
    select_fg_idx = None
    select_bg_idx = None

    def __init__(self, image, mask, path, line_size, ground_truth=None, n_features=None):
        self.image = image
        self.mask = mask
        self.path = path
        self.line_size = line_size

        if ground_truth is not None and n_features is not None:
            all_fg_idx = np.argwhere(ground_truth == 255)
            rand_fg_idx = np.random.choice(all_fg_idx.shape[0], n_features, False)
            select_fg_idx = all_fg_idx[rand_fg_idx]

            all_bg_idx = np.argwhere(ground_truth == 0)
            rand_bg_idx = np.random.choice(all_bg_idx.shape[0], n_features, False)
            select_bg_idx = all_bg_idx[rand_bg_idx]

            self.select_fg_idx = tuple(select_fg_idx.T)
            self.select_bg_idx = tuple(select_bg_idx.T)

    def get_pixel_feat(self):
        fov_data = self.image[self.mask == 255]
        norm_fov_data = normalize(fov_data).ravel()

        if self.select_fg_idx is None or self.select_bg_idx is None:
            return norm_fov_data
        else:
            norm_image = np.zeros(self.mask.shape, np.float64)
            norm_image[self.mask == 255] = norm_fov_data

            fg_feat = norm_image[self.select_fg_idx]
            bg_feat = norm_image[self.select_bg_idx]

            return fg_feat, bg_feat

    def get_gaussian_feat(self):
        kernels = [(i, i) for i in range(3, 13, 2)]
        filtered_images = [cv2.GaussianBlur(self.image, kernel, 0) for kernel in kernels]

        if self.select_fg_idx is None or self.select_bg_idx is None:
            return np.dstack(filtered_images)
        else:
            fg_feat = [image[self.select_fg_idx] for image in filtered_images]
            fg_feat = np.column_stack(fg_feat)

            bg_feat = [image[self.select_bg_idx] for image in filtered_images]
            bg_feat = np.column_stack(bg_feat)

            return fg_feat, bg_feat

    def get_gabor_feat(self):
        orientations = (np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi)
        kernels = [cv2.getGaborKernel((3, 3), 3, orientation, 5, .5) for orientation in orientations]
        filtered_images = [cv2.filter2D(self.image, -1, kernel) for kernel in kernels]

        if self.select_fg_idx is None or self.select_bg_idx is None:
            return np.dstack(filtered_images)
        else:
            fg_feat = [image[self.select_fg_idx] for image in filtered_images]
            fg_feat = np.column_stack(fg_feat)

            bg_feat = [image[self.select_bg_idx] for image in filtered_images]
            bg_feat = np.column_stack(bg_feat)

            features = np.vstack((fg_feat, bg_feat))

            return features

    def get_single_linestr_feat(self):
        window_avg = window_average.cached_basic(self.path, self.image, self.mask, self.line_size)
        linestr = line_operator.cached_single(self.path, self.image, self.mask, window_avg, self.line_size)
        fov_data = linestr[self.mask == 255]
        norm_fov_data = normalize(fov_data).ravel()

        if self.select_fg_idx is None or self.select_bg_idx is None:
            return norm_fov_data
        else:
            norm_image = np.zeros(self.mask.shape, np.float64)
            norm_image[self.mask == 255] = norm_fov_data

            fg_feat = norm_image[self.select_fg_idx]
            bg_feat = norm_image[self.select_bg_idx]

            return fg_feat, bg_feat

    def get_multi_linestr_feat(self):
        linestr = line_operator.multi(self.path, self.image, self.mask, self.line_size)
        fov_data = linestr[self.mask == 255]
        norm_fov_data = normalize(fov_data).ravel()

        if self.select_fg_idx is None or self.select_bg_idx is None:
            return norm_fov_data
        else:
            norm_image = np.zeros(self.mask.shape, np.float64)
            norm_image[self.mask == 255] = norm_fov_data

            fg_feat = norm_image[self.select_fg_idx]
            bg_feat = norm_image[self.select_bg_idx]

            return fg_feat, bg_feat
