import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes


def importance(img_size, i_space, Z_boundary, sample_size):
    """ """
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    x_axis = np.linspace(x_min, x_max, img_size)
    y_axis = np.linspace(y_min, y_max, img_size)

    sample_size = 100

    X, Y = np.meshgrid(x_axis, y_axis)
    Z = np.zeros(X.shape)

    for j, x in enumerate(x_axis):
        for i, y in enumerate(y_axis):
            Z[i, j] = mandelbrot.eval_point_mandelbrot(x, y, i_space)

    pixels = Z >= Z_boundary
    pixels_filled = binary_fill_holes(pixels)
    #importance_sample_space(pixels_filled, i)

    # Compute the area occupied by pixels/boxes.
    pixel_nr = np.sum(pixels_filled)
    dx = (x_max - x_min) / img_size
    dy = (y_max - y_min) / img_size
    pixel_area = abs(dx * dy)
    area = pixel_nr * pixel_area

    # Collects samples within importance space
    samples = []

    # random samples added if in area.
    while len(samples) < sample_size:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        # fix pixel indexes for x and y to determine if sample falls within
        x_idx = int((x - x_min) / dx)
        y_idx = int((y - y_min) / dy)

        # Check if sample is within pixel for which value is True (part of area)
        if 0 <= x_idx < img_size and 0 <= y_idx < img_size:
            if pixels_filled[y_idx, x_idx]: 
                samples.append((x, y))
    
    return area, samples


def importance_sample_space(pixels_filled, i):
    """Makes a plot of the importance sample space area."""

    plt.imshow(pixels_filled, extent=(-2, 1, -1.5, 1.5), alpha=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Mandelbrot Set Approximation (Max Iterations = {i})')
    plt.gca().set_aspect('equal')
    plt.show()