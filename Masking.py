import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes


def masking(img_size, i_space, Z_boundary, sample_size):
    """ Takes samples based on an defined area within which to accept samples."""
    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5
    x_axis = np.linspace(x_min, x_max, img_size)
    y_axis = np.linspace(y_min, y_max, img_size)

    X, Y = np.meshgrid(x_axis, y_axis)
    Z = np.zeros(X.shape)

    for j, x in enumerate(x_axis):
        for i, y in enumerate(y_axis):
            Z[i, j] = mandelbrot.eval_point_mandelbrot(x, y, i_space)

    pixels = Z >= Z_boundary
    pixels_filled = binary_fill_holes(pixels)
    #masking_sample_space(pixels_filled, i)

    # Compute the area occupied by pixels/boxes.
    pixel_nr = np.sum(pixels_filled)
    dx = (x_max - x_min) / img_size
    dy = (y_max - y_min) / img_size
    pixel_area = abs(dx * dy)
    area = pixel_nr * pixel_area


     # Set valid pixels and number of grids
    valid_pixels = np.argwhere(pixels_filled)
    num_valid_pixels = len(valid_pixels)
    samples_per_pixel = sample_size // num_valid_pixels

    x_samples = []
    y_samples = []

    for _, (i, j) in enumerate(valid_pixels):
        x_bound = x_min + j * dx
        y_bound = y_min + i * dy

        # Make sure the subgrids can hold enough samples, make larger if neccesary
        # Samples per pixel can be different due to number of grids and devision.
        subgrid_size = int(np.sqrt(samples_per_pixel))
        if subgrid_size * subgrid_size < samples_per_pixel:
            subgrid_size += 1

        x_edges = np.linspace(x_bound, x_bound + dx, subgrid_size + 1)
        y_edges = np.linspace(y_bound, y_bound + dy, subgrid_size + 1)
        
        subgrid_size = int(np.sqrt(samples_per_pixel))
        if subgrid_size * subgrid_size < samples_per_pixel:
            subgrid_size += 1

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

    # sample size may differ, so change in mb
    samples = list(zip(x_samples, y_samples))

    return area, pixels_filled, samples


def masking_sample_space(samples, area_filled, i_space, Z_boundary):
    """Makes a plot of the masking sample space area and samples."""

    sample_x, sample_y = zip(*samples)

    plt.imshow(area_filled, extent=(-2, 1, -1.5, 1.5), alpha=0.25)
    plt.scatter(sample_x, sample_y, color='red', s=5, label="Samples")
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Sampled Points with Max Iterations = {i_space} and Z boundary = {Z_boundary}')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()
