"""
Custom Masing and Orthogonal Sampling of the Mandelbrot Set

This module provides tools for a custom masking/adaptive orthogonal 
sampling method of the Mandelbrot set by generating samples orthogonally 
within a masked grid. The grid is adjusted based on the boundaries of 
the Mandelbrot set.
This module includes:

1. Adaptive Grid Sampling:
   - Constructs a masked grid to identify valid areas of the Mandelbrot 
     set.
   - Generates sample points orthogonally from each cell in the grid.

2. Visualization:
   - Visualizes the adaptive grid sample space with highlighted valid 
     regions.

Modules Imported:
- matplotlib.pyplot: For visualization of the adaptive sample space.
- numpy: For numerical computations, grid creation, and sampling.
- mandelbrot: Provides the function to evaluate whether a point lies 
              within the Mandelbrot set.

Usage:
Run 'adaptive' to calculate the area and generate sample points from 
the Mandelbrot set. Run 'adaptive_sample_space' to visualize the 
adaptive grid sample space. 
"""

import matplotlib.pyplot as plt
import numpy as np

import mandelbrot


def adaptive(img_size, i_space, z_boundary, sample_size):
    """
    Defines and determines a masked grid for sampling from the 
    Mandelbrot set. Generates samples orthogonally from each cell in the 
    masked grid.

    Parameters:
    img_size (int): Number of pixels along one dimension to determine 
        grid size.
    i_space (int): Maximum number of iterations for evaluating the 
        Mandelbrot set.
    z_boundary (float): Threshold value to accept a pixel.
    sample_size (int): Total number of samples to generate.

    Returns:
    area (float): Area of the masked grid occupied by valid pixels.
    samples (list of tuples): List of sampled points (x, y) within the 
        valid grid cells.
    """
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    x_axis = np.linspace(x_min, x_max, img_size)
    y_axis = np.linspace(y_min, y_max, img_size)
    X, Y = np.meshgrid(x_axis, y_axis)

    # Create an array to store masked area as True/False.
    Z = np.zeros(X.shape)

    # Evaluate pixels to determine if part of masked Mandelbrot area.
    for j, x in enumerate(x_axis):
        for i, y in enumerate(y_axis):
            Z[i, j] = mandelbrot.eval_point_mandelbrot(x, y, i_space)

    # Create boolean for points that meet threshhold value z_boundary.
    pixels = Z >= z_boundary
    pixels_filled = pixels

    # Compute the area occupied by valid pixels.
    pixel_nr = np.sum(pixels_filled)
    dx = (x_max - x_min) / img_size
    dy = (y_max - y_min) / img_size
    pixel_area = abs(dx * dy)
    area = pixel_nr * pixel_area

    # Determine the number of samples to take per pixel.
    valid_pixels = np.argwhere(pixels_filled)
    num_valid_pixels = len(valid_pixels)
    samples_per_pixel = sample_size // num_valid_pixels
    subgrid_size = int(np.sqrt(samples_per_pixel))

    x_samples = []
    y_samples = []

    # Perform orthogonal sampling for every valid pixel.
    for _, (i, j) in enumerate(valid_pixels):
        x_bound = x_min + j * dx
        y_bound = y_min + i * dy

        x_edges = np.linspace(x_bound, x_bound + dx, subgrid_size + 1)
        y_edges = np.linspace(y_bound, y_bound + dy, subgrid_size + 1)

        indices = np.arange(subgrid_size)

        # Make the subgrid squares.
        subgrid_squares = []
        for xi in indices:
            for yi in indices:
                subgrid_squares.append((xi, yi))

        np.random.shuffle(subgrid_squares)

        # Sample in the subgrids.
        for xi, yi in subgrid_squares:
            x_sample = np.random.uniform(x_edges[xi], x_edges[xi + 1])
            y_sample = np.random.uniform(y_edges[yi], y_edges[yi + 1])
            x_samples.append(x_sample)
            y_samples.append(y_sample)
    
    samples = list(zip(x_samples, y_samples))

    return area, samples


def adaptive_sample_space(area_filled):
    """
    Visualizes the adaptive grid sample space with valid areas 
    highlighted.
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(area_filled, extent=(-2, 1, -1.5, 1.5), alpha=0.25)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.xlabel('Real Part', fontsize=12)
    plt.ylabel('Imaginary Part', fontsize=12)
    plt.grid()
    plt.savefig('Figures/Adaptive Sample Space.pdf',
                bbox_inches='tight', format='pdf')
    plt.show()
