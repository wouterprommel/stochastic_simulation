"""
Latin Hypercube and Orthogonal Sampling Methods

This module provides sampling techniques for the Mandelbrot set, 
using Latin hypercube and orthogonal sampling methods. 
The module also demonstrates the evaluation of sampled points against 
the Mandelbrot set.

This module includes:

1. hypercube: Implements Latin hypercube sampling.
2. orthogonal: Implements orthogonal sampling on an NxN grid.

Usage:
Run the script to generate samples using the hypercube or orthogonal 
methods and evaluate them against the Mandelbrot set.
"""

import numpy as np


def hypercube(N):
    """
    Applies Latin hypercube sampling to the Mandelbrot set grid of x 
    and y values with the specified sample size N.

    Parameters:
    N (int): Number of samples to generate.

    Returns:
    samples (list of tuples) as a list of (x, y) points sampled from the 
    Mandelbrot set grid using Latin hypercube sampling.
    """
    # Configure grid dimensions.
    edges = N+1
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

    # Shuffle y samples to make sure there is no correlation.
    np.random.shuffle(y_samples)
    hypercube_samples = list(zip(x_samples, y_samples))

    return hypercube_samples


def orthogonal(N):
    """
    Applies orthogonal sampling to the Mandelbrot set grid of x and y 
    values with specified sample size N. The NxN grid must be a perfect 
    square for orthogonal sampling to function.

    Parameters:
    N (int): Number of samples to generate. Must be a perfect square.

    Returns:
    samples (list of tuples) as a list of (x, y) points sampled from the 
    Mandelbrot set grid using orthogonal sampling.

    Raises:
    ValueError: If N is not a perfect square.
    """
    # Ensure that N * N is a perfect square.
    k = int(np.sqrt(N))
    if k * k != N:
        raise ValueError(
            "NxN must be a perfect square for orthogonal sampling.")

    # Define main grid.
    edges = k+1  # Interval edges.
    x_axis = np.linspace(-2, 1, edges)
    y_axis = np.linspace(-1.5, 1.5, edges)

    x_samples = []
    y_samples = []

    # Create all index cominations and randomize order.
    squares = [(xi, yi) for xi in range(k) for yi in range(k)]
    np.random.shuffle(squares)

    # Sample randomly in each square.
    for xi, yi in squares:
        x = np.random.uniform(x_axis[xi], x_axis[xi+1])
        y = np.random.uniform(y_axis[yi], y_axis[yi+1])

        x_samples.append(x)
        y_samples.append(y)

    ortho_samples = list(zip(x_samples, y_samples))

    return ortho_samples
