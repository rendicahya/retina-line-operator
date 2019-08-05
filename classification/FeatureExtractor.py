import cv2
import numpy as np

from methods.multi_line_opr import cached_multi
from util.image_util import zero_one_norm


class FeatureExtractor:
    selected_fg_idx = None
    selected_bg_idx = None

    def __init__(self, image, mask, path, size, ground=None, n_features=None):
        self.image = image
        self.mask = mask
        self.path = path
        self.size = size

        if ground is not None and n_features is not None:
            np.random.seed(0)

            all_fg_idx = np.argwhere(ground == 255)
            rand_fg_idx = np.random.choice(all_fg_idx.shape[0], n_features // 2, False)
            selected_fg_idx = all_fg_idx[rand_fg_idx]

            np.random.seed(1)

            all_bg_idx = np.argwhere(ground == 0)
            rand_bg_idx = np.random.choice(all_bg_idx.shape[0], n_features // 2, False)
            selected_bg_idx = all_bg_idx[rand_bg_idx]

            self.selected_fg_idx = tuple(selected_fg_idx.T)
            self.selected_bg_idx = tuple(selected_bg_idx.T)

    def get_pixel_feat(self):
        fov = self.image[self.mask == 255]
        norm_fov = zero_one_norm(fov).ravel()

        if self.selected_fg_idx is None or self.selected_bg_idx is None:
            return norm_fov
        else:
            norm_image = np.zeros(self.mask.shape, np.float64)
            norm_image[self.mask == 255] = norm_fov

            fg_feat = norm_image[self.selected_fg_idx]
            bg_feat = norm_image[self.selected_bg_idx]

            return fg_feat, bg_feat

    def get_gaussian_feat(self):
        kernels = [(i, i) for i in range(3, 13, 2)]
        filtered = [cv2.GaussianBlur(self.image, kernel, 0) for kernel in kernels]

        if self.selected_fg_idx is None or self.selected_bg_idx is None:
            return np.dstack(filtered)
        else:
            fg_feat = [img[self.selected_fg_idx] for img in filtered]
            fg_feat = np.column_stack(fg_feat)

            bg_feat = [img[self.selected_bg_idx] for img in filtered]
            bg_feat = np.column_stack(bg_feat)

            return fg_feat, bg_feat

    def get_gabor_feat(self):
        orientations = (np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi)
        kernels = [cv2.getGaborKernel((3, 3), 3, orientation, 5, .5) for orientation in orientations]
        filtered_images = [cv2.filter2D(self.image, -1, kernel) for kernel in kernels]

        if self.selected_fg_idx is None or self.selected_bg_idx is None:
            return np.dstack(filtered_images)
        else:
            fg_feat = [image[self.selected_fg_idx] for image in filtered_images]
            fg_feat = np.column_stack(fg_feat)

            bg_feat = [image[self.selected_bg_idx] for image in filtered_images]
            bg_feat = np.column_stack(bg_feat)

            features = np.vstack((fg_feat, bg_feat))

            return features

    def get_single_linestr_feat(self):
        line_str = cached_multi(self.path, self.image, self.mask, self.size)
        line_str_fov = line_str[self.mask == 255]
        norm_fov_data = zero_one_norm(line_str_fov).ravel()

        if self.selected_fg_idx is None or self.selected_bg_idx is None:
            return norm_fov_data
        else:
            norm_image = np.zeros(self.mask.shape, np.float64)
            norm_image[self.mask == 255] = norm_fov_data

            fg_feat = norm_image[self.selected_fg_idx]
            bg_feat = norm_image[self.selected_bg_idx]

            return fg_feat, bg_feat

    def get_multi_linestr_feat(self):
        line_str = cached_multi(self.path, self.image, self.mask, self.size)
        line_str_fov = line_str[self.mask == 255]
        norm_fov_data = zero_one_norm(line_str_fov).ravel()

        if self.selected_fg_idx is None or self.selected_bg_idx is None:
            return norm_fov_data
        else:
            norm_image = np.zeros(self.mask.shape, np.float64)
            norm_image[self.mask == 255] = norm_fov_data

            fg_feat = norm_image[self.selected_fg_idx]
            bg_feat = norm_image[self.selected_bg_idx]

            return fg_feat, bg_feat
