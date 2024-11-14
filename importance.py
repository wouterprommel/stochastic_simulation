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

from matplotlib.path import Path
from matplotlib.patches import PathPatch

def importance_space(img_size, max_iteration):
    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = np.zeros(X.shape)

    for j, x in enumerate(x_axis):
        if j % 100 == 0: print(j)
        for i, y in enumerate(y_axis):

            Z[i, j] = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)

    area = plt.contourf(X, Y, Z, levels=np.linspace(0.95, 1.0, 5), colors='red')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Mandelbrot Set Approximation (Max Iterations = {max_iteration})')
    plt.gca().set_aspect('equal')
    plt.show()

    # Get the paths of the filled contour areas
    paths = [Path(p.vertices) for collection in area.collections for p in collection.get_paths()]

    return paths


# Find the area (check paths in plt)
def importance_area(paths):
    """Calculate the area of the filled contour Mendelbroth approximation."""
    area_total = 0

    # iterate over the paths and add areas for each path to the total
    for path in paths:
        area_part = PathPatch(path)
        area_total += area_part.get_path().area()

    return area_total


# Sample from within the area
def importance_sample(paths, sample_size):
    """Generates randomly assigned points within the area."""

    x_min, x_max = -2, 1
    y_min, y_max = -1.5, 1.5

    samples = []

    while len(samples) < sample_size:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        sample = (x, y)

        # How iterate over paths?
        # Special function in path (contains_point)
        if any(path.contains_point(sample) for path in paths):
            #append sample
            samples.append(sample)
    print("Samples: ", samples)



    #Cannot sample only in area, so points are taken randomly 
    # but added only when inside the area



    return


if __name__ == "__main__":

    img_size = 50
    max_iteration = 5

    importance_space(img_size, max_iteration)








# Approach 2:
"""
1. Look at different areas to see how they converge and what the difference is between them.

"""

