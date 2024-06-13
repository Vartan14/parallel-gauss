import numpy as np
from time import time

from algorithm.serial.default_gauss import gauss


def test_gauss(n):
    extended_matrix = np.random.uniform(0, 255, size=(n, n + 1))

    serial_timer = time()
    x_serial = gauss(extended_matrix)
    print(f"n: {n}\ttime: {round(time() - serial_timer, 4)}")


if __name__ == '__main__':
    print("Test Serial Gauss Algorithm")
    values = [50, 100, 200, 400]

    for n in values:
        test_gauss(n)
