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

def importance_space(img_size, max_iteration):
    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = np.zeros(X.shape)

    for j, x in enumerate(x_axis):
        if j % 100 == 0: print(j)
        for i, y in enumerate(y_axis):

            Z[i, j] = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)

    plt.contour(X, Y, Z, levels=np.linspace(0.95, 1.0, 5), colors='red')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title(f'Mandelbrot Set Approximation (Max Iterations = {max_iteration})')
    plt.gca().set_aspect('equal')
    plt.show()



if __name__ == "__main__":

    img_size = 500
    max_iteration = 5

    importance_space(img_size, max_iteration)








# Approach 2:
"""
1. Look at different areas to see how they converge and what the difference is between them.

"""

