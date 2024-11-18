"""
Analysis and Area Estimation of the Mandelbrot Set.

This module analyses and visualizes the Mandelbrot set.
It includes methods to:

1. Evaluate if a point lies within the Mandelbrot set using 
   iterative calculations.
2. Estimate the area of the Mandelbrot set using Monte Carlo sampling 
   methods:
   - Uniform random sampling.
   - Latin Hypercube sampling.
   - Orthogonal sampling.
   - Custom Masking and orthogonal sampling.
3. Computing the area of the Mandelbrot set using pixel counting method.
4. Timing and comparing the efficiency of different Monte Carlo sampling 
   techniques.
5. Visualizing the Mandelbrot set and sampling distributions.

Modules Imported:
- time: Timing performance of sampling methods.
- numpy: For numerical computations and random number generation.
- matplotlib.pyplot: For visualization of the Mandelbrot set and sample 
                     points.
- Sample_methods: Provides the sampling methods hypercube and 
                  orthogonal.
- Masking: Handles adaptive grid-based sampling.

Usage:
Run the module to compute and visualize the Mandelbrot set using the 
pixel counting method and compare the estimated value with different 
sampling methods that are timed.

Generates a file: python mandelbrot_analysis.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import Masking
import Sample_methods


def eval_point_mandelbrot(x, y, i):
    """
    Determine if a point lies in the Mandelbrot set.

    Parameters:
    x (float): Real part of the complex number.
    y (float): Imaginary part of the complex number.
    i (int): Maximum number of iterations.

    Returns: A float with normalized iteration count (range 0-1).
    """
    max_iteration = i
    c = complex(x, y)
    z = 0
    iteration = 0
    bounded = True
    iterating = True

    # Iterate over the point until it escapes.
    while bounded and iterating:
        bounded = abs(z) <= 2
        iterating = iteration < max_iteration
        z = z*z + c
        iteration += 1

    return (iteration-1)/max_iteration  # set red color (0-1)


def mc_area(N, i, method='uniform', std=False):
    """
    Compute the area of the Mandelbrot set using Monte Carlo methods.

    Parameters:
    N (int): Number of sample points.
    i (int): Maximum number of iterations for the Mandelbrot evaluation.
    method (str): Sampling method to use 
        ('uniform', 'hypercube', 'orthogonal', 'adaptive').
        Default is 'uniform'.
    std (bool): Whether to return the standard deviation. 
        Default is False.

    Returns: Estimated area of the Mandelbrot set as a float. 
        If `std=True`, returns a tuple (area, standard deviation).
    """
    area_total = 9
    N = int(N)

    # Generate sample points according to specified methods.
    if method == 'uniform':
        X = -1.5 + 3*np.random.rand(N)
        Y = -2 + 3*np.random.rand(N)
        samples = list(zip(X, Y))

    elif method == 'hypercube':
        samples = Sample_methods.hypercube(N)

    elif method == 'orthogonal':
        samples = Sample_methods.orthogonal(N)

    elif method == 'adaptive':
        img_size = 25
        i_space = 8
        z_boundary = 1
        area_total, samples = Masking.adaptive(
            img_size, i_space, z_boundary, N)
        N = len(samples)

    # Evaluate to determine if a sample belongs to the Mandelbrot set.
    evaluations = []
    for x, y in samples:
        evaluation = eval_point_mandelbrot(x, y, i) == 1
        evaluations.append(evaluation)

    # Calculate the estimated Mandelbrot set area and standard deviation.
    area = area_total * sum(evaluations) / len(evaluations)
    std_value = area_total * \
        np.std(evaluations, ddof=1)/np.sqrt(len(evaluations))

    if std is True:
        return area, std_value

    return area


def pixel_count_area(img_size=1000):
    """
    Compute the area of the Mandelbrot set with a pixel counting method.

    Parameters:
    img_size (int): Resolution of the grid used for pixel counting. 
        Default is 1000.

    Returns: None, displays and saves the visualization of the 
        Mandelbrot set.
    """
    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    max_iteration = 80

    # Initialize an array to an generate inferno map for the Mandelbrot set.
    mandelbrot_set = np.zeros((img_size, img_size))

    # Evaluate grid points in the array of the Mandelbrot set area
    for j, x in enumerate(x_axis):
        for i, y in enumerate(y_axis):
            mandelbrot_set[i, j] = eval_point_mandelbrot(x, y, max_iteration)

    # Count the pixels inside the Mandelbrot set.
    S = np.sum(mandelbrot_set[:, :] == 1)
    A = S/(img_size*img_size) * 3*3

    print(f"Area from pixel count: {A=}")
    print("The relative error compared to the literature value is:",
          f"{np.abs(A - 1.5065)/1.5065*100}%.")

    # Visualize the Mandelbrot set.
    plt.figure(figsize=(5.91/2, 3.6/2))
    plt.imshow(mandelbrot_set, extent=(-2, 1, -1.5, 1.5), cmap='inferno')
    colorbar = plt.colorbar()
    colorbar.ax.tick_params(labelsize=8)
    plt.xlabel('Real Part', fontsize=8)
    plt.ylabel('Imaginary Part', fontsize=8)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.savefig('Figures/Mandelbrot.pdf', bbox_inches='tight', format='pdf')
    plt.show()


def timeing():
    """
    Measure the execution time for the different Monte Carlo sampling
    methods and prints the elapsed time for each sampling method.
    """
    N = int(1e4)

    # Executes Pure random Monte Carlo sampling.
    t = time.time()
    print(mc_area(N, 80))
    print(np.round(time.time() - t, 3), 'sec elapsed for random mb')

    # Executes Latin Hypercube Monte Carlo sampling.
    t = time.time()
    print(mc_area(N, 80, 'hypercube'))
    print(np.round(time.time() - t, 3), 'sec elapsed for hypercube mb')

    # Executes Orthogonal Monte Carlo sampling.
    t = time.time()
    print(mc_area(N, 80, 'orthogonal'))
    print(np.round(time.time() - t, 3), 'sec elapsed for orthogonal mb')

    # Executes custom Masking Monte Carlo sampling.
    t = time.time()
    print(mc_area(N, 80, 'adaptive'))
    print(np.round(time.time() - t, 3), 
          'sec elapsed for masking orthogonal mb')


def plot_samples():
    """Plots the sample points for different sampling methods. """
    N = int(25)

    samples = Sample_methods.hypercube(N)
    samples2 = Sample_methods.orthogonal(N)

    plt.scatter(*zip(*samples))
    plt.scatter(*zip(*samples2))
    plt.show()


if __name__ == "__main__":
    # Run pixel counting and timeing analysis.
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    pixel_count_area()
    timeing()
