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
    paths = [Path(p.vertices) for collection in paths.collections for p in collection.get_paths()]

    return paths



    # Define the bounds of the area of interest
    interest_area_indices = np.argwhere((Z >= 0.95) & (Z <= 1.0))
    x_interest_min, x_interest_max = x_axis[interest_area_indices[:, 1].min()], x_axis[interest_area_indices[:, 1].max()]
    y_interest_min, y_interest_max = y_axis[interest_area_indices[:, 0].min()], y_axis[interest_area_indices[:, 0].max()]

    print("x_interest_min: ", x_interest_min)
    print("x_interest_max: ", x_interest_max)
    print("y_interest_min: ", y_interest_min)
    print("y_interest_max: ", y_interest_max)




if __name__ == "__main__":

    img_size = 50
    max_iteration = 5

    importance_space(img_size, max_iteration)








# Approach 2:
"""
1. Look at different areas to see how they converge and what the difference is between them.

"""

