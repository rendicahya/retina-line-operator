from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.image_util import subtract_masked, normalize_masked


def proposed_norm(path, img, mask, size):
    window = cached_integral(path, img, mask, size)
    stat = cached_statistics(path, img, mask, size)
    sub = subtract_masked(stat['min'], window, mask)
    min_window = normalize_masked(sub, mask)

    return min_window
