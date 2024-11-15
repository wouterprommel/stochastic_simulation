'''
Checks:
1. Check if pixelarea correct (checked)
2. Check if 100 pixels are looped over for Z (checked and correct)
3. Check id pixels and pixels filled correct (correct)
4. check if area calc correct (checked)
'''

import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes


def importance(img_size, i_space, Z_boundary, sample_size):
    """ 
    Generates a space within which to sample based on maximum iterations and Z boundary.
    Calculates the area of the space.
    Randomly generates points and adds them to samples if they are within the predefined area.
    Larger image size means more pixels and more accurate sample space.
    higher iterations is more accurate and complex sample space."""

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


    # Generates a list of pixels/boxes where Z values >= the Z boundary is set to True
    # Fills in gaps within the area so that it is enclosed
    pixels = Z >= Z_boundary
    pixels_filled = binary_fill_holes(pixels)

    # Compute the area occupied by pixels/boxes.
    pixel_nr = np.sum(pixels_filled)
    dx = (x_max - x_min) / img_size
    dy = (y_max - y_min) / img_size
    pixel_area = abs(dx * dy)
    area = pixel_nr * pixel_area


    ''' plt.imshow(pixels_filled, extent=(-2, 1, -1.5, 1.5), alpha=0.5)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Mandelbrot Set Approximation (Max Iterations = {max_iteration})')
    plt.gca().set_aspect('equal')
    plt.show()'''


    # Collects samples within importance space
    samples = []

    # Cannot sample only in area, so points are taken randomly, 
    # but added only when inside the area.
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


'''if __name__ == "__main__":

    img_size = 30
    i_space = 10
    sample_size = 100
    i = 80

    # Boundary for which to accept area pixels/boxes
    Z_boundary = 0.95

    area_filled = importance(img_size, i_space, Z_boundary, sample_size)
    #area = importance_area(area_filled, img_size)
    #samples = importance_sample_random(area_filled, sample_size, img_size)
    #print(importance_mb_area(area_filled, sample_size, img_size, i, area, std=False))

    sample_x, sample_y = zip(*samples)

    plt.imshow(area_filled, extent=(-2, 1, -1.5, 1.5), alpha=0.25)
    plt.scatter(sample_x, sample_y, color='red', s=5, label="Samples")
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Sampled Points within Mandelbrot Set (Max Iterations = {max_iteration})')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()'''



