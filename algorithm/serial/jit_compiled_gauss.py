import numpy as np
from numba import jit, prange, NumbaPerformanceWarning
import time
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@jit(nopython=False, parallel=False)
def numba_jit_gauss(ext_a):
    n = len(ext_a)
    x = np.zeros(n)

    # Check if matrix is square
    if ext_a.shape[0] != ext_a.shape[1] - 1:
        raise ValueError("Matrix must be square!")

    # Elimination iteration
    for k in range(n - 1):
        if np.fabs(ext_a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(ext_a[i, k]) > np.fabs(ext_a[k, k]):
                    for j in range(k, n + 1):
                        ext_a[k, j], ext_a[i, j] = ext_a[i, j], ext_a[k, j]
                    break

        for i in prange(k + 1, n):
            if ext_a[i, k] != 0:
                factor = ext_a[k, k] / ext_a[i, k]
                for j in range(k, n + 1):
                    ext_a[i, j] = ext_a[k, j] - factor * ext_a[i, j]

    # Back substitution
    if ext_a[n - 1, n - 1] == 0:
        raise ValueError('System does not have solutions.')

    x[n - 1] = ext_a[n - 1, n] / ext_a[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        if ext_a[i, i] == 0:
            raise ValueError(f'System does not have solutions')

        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += ext_a[i, j] * x[j]

        x[i] = (ext_a[i, n] - sum_ax) / ext_a[i, i]

    return x


def read_matrix(filename: str) -> np.ndarray[np.float64] | None:
    with open(filename, 'r') as f:
        ext_matrix = np.loadtxt(f)

    return ext_matrix


if __name__ == '__main__':
    N = 1000
    extended_matrix = np.random.uniform(1, 10, size=(N, N+1))
    # extended_matrix = read_matrix("../../data/debug/matrix 8x8.txt")
    # print(f"Extended matrix A:\n{extended_matrix}")

    start_time = time.time()
    x = numba_jit_gauss(extended_matrix)
    print(f"\nx:\n{x[:10]}")
    print(f"Time: {time.time() - start_time}")
