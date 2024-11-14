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
from scipy.spatial import ConvexHull

def importance_space(img_size, max_iteration):
    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    sample_space = []
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = np.zeros(X.shape)

    for j, x in enumerate(x_axis):
        if j % 100 == 0: print(j)
        for i, y in enumerate(y_axis):

            # To generate inferno map
            Z[i, j] = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)

            if eval == 1:
                sample_space.append([j, i])
            
    x_vals = [point[0] for point in sample_space]
    y_vals = [point[1] for point in sample_space]

    #plt.scatter(x_vals, y_vals, s=0.5)

    plt.imshow(Z, extent=(-2, 1, -1.5, 1.5), origin='lower', cmap='inferno')
    
    plt.colorbar()
    plt.title('Sample space')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.show()
    print("Z: ", Z)

    return


if __name__ == "__main__":

    importance_space(1000, 10)








# Approach 2:
"""
1. Look at different areas to see how they converge and what the difference is between them.

"""

