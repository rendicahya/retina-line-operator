from coba.module1 import add


def run(op):
    a = 10
    b = op(a)

    print(b)


if __name__ == '__main__':
    run(add)
