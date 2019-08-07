import numpy as np


def to_numpy_array(*args):
    arr = [item[:] if type(item) is np.ndarray else np.array(item) for item in args]

    return arr[0] if len(args) == 1 else arr


if __name__ == '__main__':
    a = [1, 2, 3]
    A = to_numpy_array(a)

    print(type(a))
    print(type(A))

    print('---')

    b = [4, 5, 6]
    A, B = to_numpy_array(a, b)

    print(type(a))
    print(type(b))
    print(type(A))
    print(type(B))
