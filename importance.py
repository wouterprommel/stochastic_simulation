# Approach 1: 
""" 
1. sample area (hypercube/stochastically)
2. Define area which to sample from (such as where iterations > 1)
    - bepaal cutoff
    - donut distribution?
3. Perform sampling as usual
    4. only sample from sample space
        - how to define and check?
        - come up with sample from sample space or check after getting random sample?
"""

import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes


def importance_space(img_size, max_iteration, Z_boundary):
    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = np.zeros(X.shape)

    for j, x in enumerate(x_axis):
        if j % 100 == 0: print(j)
        for i, y in enumerate(y_axis):
            Z[i, j] = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)
    
    area = Z >= Z_boundary
    area_filled = binary_fill_holes(area)

    #area_count = np.sum(area_filled)
    #print("area count: ", area_count)
    
    # Check if it works
    #truth = np.sum(area_filled)
    #print("truth: ", truth)

    plt.imshow(area_filled, extent=(-2, 1, -1.5, 1.5), alpha=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Mandelbrot Set Approximation (Max Iterations = {max_iteration})')
    plt.gca().set_aspect('equal')
    plt.show()

    return area_filled


# Find the area (check paths in plt)
def importance_area(area_filled, img_size):
    """Calculate the area of the filled contour Mendelbroth approximation."""
    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)

    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]

    pixel_area = dx * dy

    # Calculate total area of the filled mask
    area = np.sum(area_filled) * pixel_area
    
    print("Total area: ", area)

    return area


# Sample from within the area
#Cannot sample only in area, so points are taken randomly 
# but added only when inside the area
def importance_sample_random(area, sample_size):
    """Generates randomly assigned points within the area."""

    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5

    samples = []

    while len(samples) < sample_size:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        sample = (x, y)

        #If the sample is within area 
        # append the sample to samples

    return samples


def importance_stdev(samples, i, area, std=False):
    evaluations = []
    for x, y in samples:
        eval = mandelbrot.eval_point_mandelbrot(x, y, i) == 1
        evaluations.append(eval)

    area_total = (importance_area(area)) * sum(evaluations) / len(evaluations)
    print(f"Area from MC: {area_total=}")
    std_value = (importance_area(area)) * np.std(evaluations, ddof=1)/np.sqrt(len(evaluations)) # sample variance

    if std == True:
        return area, std_value
    else: 
        return area


if __name__ == "__main__":

    img_size = 100
    max_iteration = 5
    sample_size = 1000
    i = 1000

    area_filled = importance_space(img_size, max_iteration, 0.95)
    #print("area filled: ", area_filled)
    importance_area(area_filled, img_size)

    #samples = importance_sample_random(paths, sample_size)

    #importance_stdev(samples, i, paths, std=False)


