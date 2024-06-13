import numpy as np
from scipy.linalg import solve
from time import time
from numba.core.errors import NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from algorithm.serial.default_gauss import gauss
from algorithm.parallel.parallel_cuda_gauss import parallel_gauss_cuda
from algorithm.serial.jit_compiled_gauss import numba_jit_gauss


def main(n: int = 100, scipy=True, serial=True, parallel=True, numba=True, threads: int = 32):
    extended_matrix = np.random.uniform(0, 255, size=(n, n + 1))
    extended_matrix_cp_1 = extended_matrix.copy()
    extended_matrix_cp_2 = extended_matrix.copy()
    extended_matrix_cp_3 = extended_matrix.copy()

    A, b = extended_matrix[:, :-1], extended_matrix[:, -1]
    #print(f"Matrix:\n{extended_matrix[:5][:5]}")

    print(f"\nTBP: {threads}")

    if scipy:
        scipy_timer = time()
        x_scipy = solve(A, b)
        print(f"\nScipy time: {time() - scipy_timer}\nx: {x_scipy[:10]}")

    if serial:
        serial_timer = time()
        x_serial = gauss(extended_matrix_cp_1)
        print(f"Serial time: {time() - serial_timer}")
        #print(f"x: {x_serial[:10]}")

    if numba:
        numba_git = time()
        x_parallel = numba_jit_gauss(extended_matrix_cp_2)
        print(f"Serial JIT time: {time() - numba_git}")
        #print(f"x:{x_parallel[:10]}")

    if parallel:
        parallel_timer = time()
        x_parallel = parallel_gauss_cuda(extended_matrix_cp_3, threads)
        print(f"Parallel JIT time: {time() - parallel_timer}")
        #print(f"x:{x_parallel[:10]}")


if __name__ == '__main__':
    for i in range(10):
        main(n=2000, scipy=False, serial=False, numba=True, threads=32)
