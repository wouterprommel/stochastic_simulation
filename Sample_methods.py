'''
This module containts the functions for the Latin hypercube sampling method
and for the orthogonal sampling method.
'''
import numpy as np

def hypercube(N):
    """
    Applies Latin hypercube sampling to the mandelbroth set grid of x and y values 
    with supplied sample size N.
    return: list
    """
    # Configure Chess board lists from dimensions.
    edges = N+1 # N intervals require N+1 edges.
    x_axis = np.linspace(-2, 1, edges)
    y_axis = np.linspace(-1.5, 1.5, edges)

    x_samples = []
    y_samples = []

    # Assigns x and y value to samples from list of options.
    for i in range(N):
        x = np.random.uniform(x_axis[i], x_axis[i+1])
        y = np.random.uniform(y_axis[i], y_axis[i+1])

        x_samples.append(x)
        y_samples.append(y)

    # Shuffle y samples to make sure there is no correlation
    np.random.shuffle(y_samples)
    hypercube_samples = list(zip(x_samples, y_samples))

    return hypercube_samples
 
def orthogonal(N):
    """
    Applies Orthogonal sampling to the mandelbroth set grid of x and y values 
    with supplied sample size N. the NxN grid must be a perfect square for the orthogonal sampling to work.
    return: list
    """
    k = int(np.sqrt(N))
    if k * k != N:
        raise ValueError("NxN must be a perfect square for orthogonal sampling.")
    # Define main grid.
    edges = k+1  # Interval edges.
    x_axis = np.linspace(-2, 1, edges)
    y_axis = np.linspace(-1.5, 1.5, edges)

    x_samples = []
    y_samples = []

    # Create all index cominations and make sure they are in random order.
    squares = [(xi, yi) for xi in range(k) for yi in range(k)]
    np.random.shuffle(squares)

    # Sampling ==> choose random point in each square and store them in samples.
    for xi, yi in squares:
        x = np.random.uniform(x_axis[xi], x_axis[xi+1])
        y = np.random.uniform(y_axis[yi], y_axis[yi+1])

        x_samples.append(x)
        y_samples.append(y)
    ortho_samples = list(zip(x_samples, y_samples))
    return ortho_samples
