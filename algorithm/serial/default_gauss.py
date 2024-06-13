import time
import numpy as np


def gauss(ext_a: np.ndarray, print_log: bool = False) -> np.ndarray:
    """
    The serial of gauss algorithm that consist of 2 steps:
    1. Forward elimination
    2. Backward substitution
    """
    n = len(ext_a)

    # Check if matrix is square
    if ext_a.shape[0] != ext_a.shape[1] - 1:
        raise ValueError("Matrix must be square!")

    # 1 Step. Forward Elimination
    forward_elimination(ext_a, n, print_log)

    # 2 Step. Backward Substitution
    return back_substitution(ext_a, n)


def forward_elimination(ext_a: np.ndarray, n: int, print_log: bool = False) -> None:
    """
    Forward elimination transforms matrix to upper right form using elementary row operations:
    - Swapping two rows
    - Multiplying a row by a nonzero number
    - Adding a multiple of one row to another row
    """

    # Elimination iteration
    for k in range(n - 1):
        # If the main diagonal element is 0 or close to 0
        if np.fabs(ext_a[k, k] < 1.0e-12):
            # Loop through rows under main diagonal element
            for i in range(k + 1, n):
                # If the el is bigger than the main diag el
                if np.fabs(ext_a[i, k]) > np.fabs(ext_a[k, k]):
                    # Then swap row i and k
                    ext_a[[k, i]] = ext_a[[i, k]]
                    if print_log:
                        print(f"Replaced rows:\n{ext_a}")
                    break

        # Loop through rows under main diagonal element
        for i in range(k + 1, n):
            # If the element under main diagonal is already 0
            if ext_a[i, k] == 0:
                continue

            factor = ext_a[k, k] / ext_a[i, k]
            # Loop each element in the row
            for j in range(k, n + 1):
                # Modify j row
                ext_a[i, j] = ext_a[k, j] - factor * ext_a[i, j]

            if print_log:
                # Log eliminate iteration
                print(f"\nk={k}, i={i}")
                print(f"factor={factor: .2f}")
                print(f"row: [", end='')
                for el in ext_a[i]:
                    print(f'\t{el:.2f}', end='')
                print(f"]")

    if print_log:
        print(f"\nTriangle matrix:\n{ext_a}")


def back_substitution(ext_a: np.ndarray, n: int) -> np.ndarray:
    """
    Back substitution calculates the values of the unknowns, starting from the last equation,
    """

    # Initialize solution vector
    x = np.zeros(n, float)

    # The x(n) = b(n) / a(nn)
    if ext_a[n - 1, n - 1] == 0:
        raise ValueError('System does not have solutions.')

    x[n - 1] = ext_a[n - 1, n] / ext_a[n - 1, n - 1]

    # Loop rows
    for i in range(n - 2, -1, -1):
        sum_ax = 0
        # Loop columns
        for j in range(i + 1, n):
            sum_ax += ext_a[i, j] * x[j]

        if ext_a[i, i] == 0:
            raise ValueError('System does not have solutions.')

        # Find x[i] in row [i]
        x[i] = (ext_a[i, n] - sum_ax) / ext_a[i, i]

    return x


def read_matrix(filename: str) -> np.ndarray[np.float64] | None:
    """
    Read matrix from file and return numpy array
    """
    with open(filename, 'r') as f:
        ext_matrix = np.loadtxt(f, dtype=np.float64)

    return ext_matrix


if __name__ == "__main__":
    ext_a_matrix = read_matrix("../../data/test/300.txt")


    #print(f"Extended matrix A:\n{ext_a_matrix}")

    start_time = time.time()
    x_vector = gauss(ext_a_matrix, False)
    print(f"x:\n{x_vector}")
    print(f"Time: {time.time() - start_time}")
