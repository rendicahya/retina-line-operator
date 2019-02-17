import cv2
import numpy as np
from bresenham import bresenham


def generate_lines(size):
    half = size // 2
    one_sixth = size // 6
    two_sixth = one_sixth * 2

    return (
               tuple(bresenham(-half, 0, half, 0)),
               tuple(bresenham(-half, -one_sixth, half, one_sixth)),
               tuple(bresenham(-half, -two_sixth, half, two_sixth)),
               tuple(bresenham(-half, -half, half, half)),
               tuple(bresenham(-two_sixth, -half, two_sixth, half)),
               tuple(bresenham(-one_sixth, -half, one_sixth, half)),
               tuple(bresenham(0, -half, 0, half)),
               tuple(bresenham(one_sixth, -half, -one_sixth, half)),
               tuple(bresenham(two_sixth, -half, -two_sixth, half)),
               tuple(bresenham(half, -half, -half, half)),
               tuple(bresenham(half, -two_sixth, -half, two_sixth)),
               tuple(bresenham(half, -one_sixth, -half, one_sixth))
           ), (
               ((0, -half), (0, half)),
               ((one_sixth, -half), (-one_sixth, half)),
               ((two_sixth, -half), (-two_sixth, half)),
               ((half, -half), (-half, half)),
               ((half, -two_sixth), (-half, two_sixth)),
               ((half, -one_sixth), (-half, one_sixth)),
               ((half, 0), (-half, 0)),
               ((half, one_sixth), (-half, -one_sixth)),
               ((half, two_sixth), (-half, -two_sixth)),
               ((half, half), (-half, -half)),
               ((two_sixth, half), (-two_sixth, -half)),
               ((one_sixth, half), (-one_sixth, -half))
           )


def generate_line_images(size, angle):
    lines, wings = generate_lines(size)
    half = size // 2
    image = np.zeros((size, size), np.uint8)

    for x, y in lines[angle]:
        image[y + half, x + half] = 255

    return image


def print_line(size):
    lines, wings = generate_lines(size)
    half = size // 2

    for i in range(len(lines)):
        for y in range(-half, half + 1):
            line_str = ''

            for x in range(-half, half + 1):
                line_str += '*' if (x, y) in lines[i] or (x, y) in wings[i] else ' '

            print(line_str)
        print('-' * size)


def main():
    print_line(9)
    image = generate_line_images(91, 1)

    cv2.imshow('Line', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
