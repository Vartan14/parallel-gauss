## Parallel and Sequential Gaussian Elimination in Python

This Python project implements both sequential and parallel versions of the Gaussian elimination algorithm for solving systems of linear equations. It allows you to compare their performance and understand the benefits of parallelization.

### Installation

1. **Clone the repository:**

   ```markdown
   git clone https://github.com/Vartan14/parallel-gauss
   ```

2. **Install dependencies:**

   The project requires the following Python libraries:

   - `numpy` (numerical computing)
   - `numba` (JIT compiler for using CUDA parallelization)

   You can install them using pip:

   ```markdown
   pip install requirements.txt
   ```

### Usage

1. **Navigate to the project directory:**

   ```markdown
   cd parallel-gauss
   ```

2. **Run the script:**

   ```markdown
   python main.py [matrix_size] [n_processes]
   ```

   - `matrix_size` (optional): The size of the square matrix to be solved. Defaults to 100.
   - `n_processes` (optional, only for parallel version): The number of processes to use for parallel execution. Defaults to the number of CPU cores on your system.

**Example:**

```markdown
python main.py 200 4  # Solve a 200x200 matrix using 4 processes
```

### Output

The script will print the following information:

- The solved upper triangular matrix (U)
- The solution vector (x)
- Execution time for both sequential and parallel versions (if using multiprocessing)

### Performance Comparison

The script displays the execution times of both sequential and parallel versions, allowing you to compare their performance. This is particularly insightful for larger matrices where parallelization can significantly reduce computation time.

![image](https://github.com/Vartan14/parallel-gauss/assets/89924019/780eae68-79bd-4050-aa96-bb6d83918dae)

