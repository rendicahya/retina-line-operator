def test(op):
    a = [1, 2, 3]
    b = op(a)

    print(a)
    print(b)


def op2(a):
    a *= 2

    return a


if __name__ == '__main__':
    test(op2)
