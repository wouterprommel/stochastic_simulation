import mandelbrot
import matplotlib.pyplot as plt
import numpy as np

# Rooks in chess

def hypercube(N):
    """
    Applies Latin hypercube sampling to the mandelbroth set grid of x and y values 
    with supplied sample size N.
    """
    # Configure Chess board lists from dimensions
    edges = N+1 # N intervals require N+1 edges 
    x_axis = np.linspace(-2, 1, edges)
    y_axis = np.linspace(-1.5, 1.5, edges)

    x_samples = []
    y_samples = []

    # Assigns x and y value to samples from list of options 
    for i in range(N):
        x = np.random.uniform(x_axis[i], x_axis[i+1])
        y = np.random.uniform(y_axis[i], y_axis[i+1])

        x_samples.append(x)
        y_samples.append(y)

    # Shuffle y samples to make sure there is no correlation
    np.random.shuffle(y_samples)
    samples = list(zip(x_samples, y_samples))

    return samples
 
def orthogonal(N):
    """
    Applies Orthogonal sampling to the mandelbroth set grid of x and y values 
    with supplied sample size N. the NxN grid must be a perfect square for the orthogonal sampling to work.
    """
    k = int(np.sqrt(N))
    if k * k != N:
        raise ValueError("NxN must be a perfect square for orthogonal sampling.")
    # Define main grid
    edges = k+1  # Interval edges
    x_axis = np.linspace(-2, 1, edges)
    y_axis = np.linspace(-1.5, 1.5, edges)

    # Define the interval indices of the subgrids
    x_indices = np.arange(k)
    y_indices = np.arange(k)

    x_samples = []
    y_samples = []
    # Create all index cominations and make sure they are in random order
    squares = [(xi, yi) for xi in range(k) for yi in range(k)]
    np.random.shuffle(squares)

    # Sampling ==> choose random point in each square and store them in samples
    for xi, yi in squares:
        x = np.random.uniform(x_axis[xi], x_axis[xi+1])
        y = np.random.uniform(y_axis[xi], y_axis[xi+1])

        x_samples.append(x)
        y_samples.append(y)
    samples = list(zip(x_samples, y_samples))
    return samples

N=9
samples = hypercube(N)
# Supply samples to evaluate mandelbroth
max_iteration = 50
evaluations_hyper = []

# Generates list of mandelbroth evaluations, that is returned (for now)
print("for the hypercube case: \n")
for x, y in samples:
    evaluation_hyper = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)
    evaluations_hyper.append(evaluation_hyper)
    print(f"Evaluated point ({x}, {y}): {evaluation_hyper}")

# Supply samples to evaluate mandelbroth
evaluations_ortho = []
samples = orthogonal(N)
# Generates list of mandelbroth evaluations
print("\nfor the orthogonal case:\n")
for x, y in samples:
    evaluation_ortho = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)
    evaluations_ortho.append(evaluation_ortho)
    print(f"Evaluated point ({x}, {y}): {evaluation_ortho}")