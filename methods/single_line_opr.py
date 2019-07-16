import cv2

from dataset.DriveDatasetLoader import DriveDatasetLoader
from methods.statistical_line_opr import cached_statistics
from methods.window_average import cached_integral
from util.image_util import normalize_masked, subtract_masked
from util.timer import Timer


def cached_single(path, img, mask, size):
    window_avg = cached_integral(path, img, mask, size)
    line_img = cached_statistics(path, img, mask, size)['max']

    return subtract_masked(line_img, window_avg, mask)


def cached_single_norm(path, img, mask, size):
    line_str = cached_single(path, img, mask, size)

    return normalize_masked(line_str, mask)


'''
def cached_line(path, img, mask, size):
    cache_dir = os.path.dirname(path) + '/cache'

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    file_path = '%s/line-%s-%d.bin' % (cache_dir, os.path.basename(path), size)

    if os.path.exists(file_path):
        binary_file = open(file_path, mode='rb')
        line_str = pickle.load(binary_file)
    else:
        line_str = line(img, mask, size)
        binary_file = open(file_path, mode='wb')

        pickle.dump(line_str, binary_file)

    binary_file.close()

    return line_str


def line(img, mask, size):
    bool_mask = mask.astype(np.bool)
    lines, wings = generate_lines(size)

    queue = mp.Queue()
    cpu_count = psutil.cpu_count()

    processes = [
        mp.Process(target=line_worker, args=(img, bool_mask, lines, queue, cpu_count, cpu_id))
        for cpu_id in range(cpu_count)]

    for p in processes:
        p.start()

    slices = [queue.get() for _ in processes]
    slices = sorted(slices, key=lambda s: s[0])
    slices = [piece[1] for piece in slices]

    for p in processes:
        p.join()

    return np.vstack(slices)
    
    
def line_worker(img, bool_mask, lines, queue, cpu_count, cpu_id):
    height, width = img.shape[:2]
    slice_height = height // cpu_count
    y_start = cpu_id * slice_height
    line_str = np.zeros((slice_height, width), np.float64)

    for Y in range(y_start, (cpu_id + 1) * slice_height):
        for X in range(width):
            if not bool_mask[Y, X]:
                continue

            max_line_avg = -sys.maxsize - 1

            for line in lines:
                line_count = 0
                line_sum = 0

                for pixel in line:
                    x = X + pixel[0]
                    y = Y + pixel[1]

                    if x < 0 or x >= width or y < 0 or y >= height or not bool_mask[y, x]:
                        continue

                    line_count += 1
                    line_sum += img[y, x]

                if line_count == 0:
                    continue

                line_avg = line_sum / line_count
                max_line_avg = max(line_avg, max_line_avg)

            line_str[Y - y_start, X] = max_line_avg

    queue.put((cpu_id, line_str))


def save_cache():
    time = Timer()

    for path, img, mask, ground_truth in DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_testing():
        img = 255 - img[:, :, 1]

        time.start(f'Window: {path} [15]')
        cached_integral(path, img, mask, 15)
        time.stop()

        for size in range(1, 16, 2):
            time.start(f'Line: {path} [{size}]')
            cached_line(path, img, mask, size)
            time.stop()
'''


def main():
    path, img, mask, ground_truth = DriveDatasetLoader('D:/Datasets/DRIVE', 10).load_training_one(1)

    img = 255 - img[:, :, 1]
    size = 15
    timer = Timer()

    timer.start('Window')
    window = cached_integral(path, img, mask, size)
    timer.stop()

    timer.start('Single')
    line_str = cached_statistics(path, img, mask, size)['max']
    line_str = subtract_masked(line_str, window, mask)
    line_str = normalize_masked(line_str, mask)
    # bin = cv2.threshold(line_str, 65, 255, cv2.THRESH_BINARY)[1]
    timer.stop()

    # timer.start('Single scale + wing')
    # single_scale_wing = single(img, mask, window_avg, size)
    # timer.finish()

    # timer.start('Find best threshold')
    # best_single_thresh, best_single = find_best_threshold(line_str, mask, ground_truth)
    # timer.finish()

    # timer.start('Multi scale')
    # multi_scale = cached_multi(path, img, mask, size)
    # timer.finish()

    # timer.start('Find best multi scale')
    # best_multi_thresh, best_multi = find_best_threshold(multi_scale, mask, ground_truth)
    # timer.finish()

    # green('Best single scale threshold: %d' % best_single_thresh)
    # green('Best multi scale threshold: %d' % best_multi_thresh)

    cv2.imshow('Image', img)
    # cv2.imshow('Window average', normalize_masked(window_avg, mask))
    cv2.imshow('Single', line_str)
    # cv2.imshow('Single + wing', normalize_masked(255 - single_scale_wing, mask))
    # cv2.imshow('Single best', 255 - normalize_masked(best_single, mask))
    # cv2.imshow('Multi', normalize_masked(multi_scale, mask))
    # cv2.imshow('Best multi', 255 - normalize_masked(best_multi, mask))
    # cv2.imshow('Multi histeq', cv2.equalizeHist(multi_scale))
    # cv2.imshow('Ground truth', ground_truth)
    # cv2.imshow('Binary', bin)
    # cv2.imshow('Multi', normalized_masked(multi_scale_norm, mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
