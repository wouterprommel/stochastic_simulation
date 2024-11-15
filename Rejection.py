import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes
import time
import pandas as pd


def rejection(img_size, i_space, Z_boundary, sample_size):
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
    #rejection_sample_space(pixels_filled, i)

    # Compute the area occupied by pixels/boxes.
    pixel_nr = np.sum(pixels_filled)
    dx = (x_max - x_min) / img_size
    dy = (y_max - y_min) / img_size
    pixel_area = abs(dx * dy)
    area = pixel_nr * pixel_area

    #samples = samples_only(sample_size, img_size, pixels_filled)
    samples = samples_reject(sample_size, img_size, pixels_filled)

    return area, pixels_filled, samples

def samples_reject(sample_size, img_size, pixels_filled):
    """Reject samples if outside of sample space."""
    samples = []

    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5

    dx = (x_max - x_min) / img_size
    dy = (y_max - y_min) / img_size

    # random samples added if
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
    #print("samples: ", samples)

    return samples


def samples_only(sample_size, img_size, pixels_filled):
    """Only get samples within the sample space."""
    samples = []

    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5

    dx = (x_max - x_min) / img_size
    dy = (y_max - y_min) / img_size


    # Collect samples within rejection space.
    valid_pixels = np.argwhere(pixels_filled)
    valid_centers = []

    # determine pixel centers for valid pixels.
    for i, j in valid_pixels:
        valid_centers.append((x_min + (j + 0.5)*dx, y_min + (i + 0.5) *dy))


    while len(samples) < sample_size:
        center_x, center_y = valid_centers[np.random.choice(len(valid_centers))]
        
        # Take random sample within center boundaries.
        x = np.random.uniform(center_x - dx / 2, center_x + dx / 2)
        y = np.random.uniform(center_y - dy / 2, center_y + dy / 2)
        
        samples.append((x, y))
    return samples


def rejection_sample_space(samples, area_filled, i_space, Z_boundary):
    """Makes a plot of the rejection sample space area and samples."""

    sample_x, sample_y = zip(*samples)

    plt.imshow(area_filled, extent=(-2, 1, -1.5, 1.5), alpha=0.25)
    plt.scatter(sample_x, sample_y, color='red', s=5, label="Samples")
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Sampled Points with Max Iterations = {i_space} and Z boundary = {Z_boundary}')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()


'''
img_size = 100
i_space = 8
Z_boundary = 0.8
sample_size = 1000
i = 80

area_total, area_filled, samples = rejection(img_size, i_space, Z_boundary, sample_size)
print("area_total: ", area_total)

evaluations = []
for x, y in samples:
    eval = eval_point_mandelbrot(x, y, i) == 1
    evaluations.append(eval)

area = area_total * sum(evaluations) / len(evaluations)
std_value = area_total * np.std(evaluations, ddof=1)/np.sqrt(len(evaluations)) # sample variance
print(f"Area from MC: {area=}, STD: {std_value}")

rejection_sample_space(samples, area_filled, i_space, Z_boundary)'''