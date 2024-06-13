import numpy as np
from numba import cuda, NumbaPerformanceWarning
import time
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


@cuda.jit
def eliminate_kernel(ext_a, n, k):
    # CUDA thread index
    i = cuda.grid(1)

    if k < i < n:
        if ext_a[i, k] != 0:
            factor = ext_a[k, k] / ext_a[i, k]
            for j in range(k, n + 1):
                ext_a[i, j] = ext_a[k, j] - factor * ext_a[i, j]


def parallel_gauss_cuda(ext_a: np.ndarray, threads_per_block: int = 32) -> np.ndarray:
    n = ext_a.shape[0]
    x = np.zeros(n)

    # Check if matrix is square
    if ext_a.shape[0] != ext_a.shape[1] - 1:
        raise ValueError("Matrix must be square!")

    # Find blocks number in a grid
    blocks_per_grid = (n + (threads_per_block - 1)) // threads_per_block

    # Copy data to device memory (GPU)
    d_ext_a = cuda.to_device(ext_a)

    # Forward Elimination
    for k in range(n - 1):
        if np.fabs(ext_a[k, k]) < 1.0e-12:
            for i in range(k + 1, n):
                if np.fabs(d_ext_a[i, k]) > np.fabs(d_ext_a[k, k]):
                    ext_a[k, :], ext_a[i, :] = ext_a[i, :], ext_a[k, :]
                    break

        # Parallel transform elements under diagonal element [k][k] to zeros
        eliminate_kernel[blocks_per_grid, threads_per_block](d_ext_a, n, k)

    # Get back upper triangle matrix to the host (CPU)
    ext_a = d_ext_a.copy_to_host()

    # Back substitution
    if ext_a[n - 1, n - 1] == 0:
        raise ValueError('System does not have solutions.')

    x[n - 1] = ext_a[n - 1, n] / ext_a[n - 1, n - 1]

    for i in range(n - 2, -1, -1):
        sum_ax = 0
        for j in range(i + 1, n):
            sum_ax += ext_a[i, j] * x[j]

        if ext_a[i, i] == 0:
            raise ValueError('System does not have solutions.')

        x[i] = (ext_a[i, n] - sum_ax) / ext_a[i, i]

    return x


if __name__ == '__main__':
    n = 1500
    #extended_matrix = read_matrix("../../data/debug/matrix 3x3.txt")
    #extended_matrix = read_matrix(f"../../data/test/{n}.txt")
    extended_matrix = np.random.uniform(0, 10, size=(n, n + 1))

    #print(f"Extended matrix A:\n{extended_matrix}")
    print(f"n: {n}")
    start_time = time.time()
    x_ = parallel_gauss_cuda(extended_matrix, 1024)
    #print(f"\nx:\n{x_[:20]}")
    print(f"Time: {time.time() - start_time}")

    start_time = time.time()
    x_ = parallel_gauss_cuda(extended_matrix, 1024)
    #print(f"\nx:\n{x_[:20]}")
    print(f"Time: {time.time() - start_time}")

