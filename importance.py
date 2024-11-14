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
    sample_space = []

    for j, x in enumerate(x_axis):
        if j % 100 == 0: print(j)
        for i, y in enumerate(y_axis):
            eval = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)
            #print("eval: ", eval)

            # Add to sample list if eval == 1
            if eval == 1:
                sample_space.append([j, i])
            
    x_vals = [point[0] for point in sample_space]
    y_vals = [point[1] for point in sample_space]

    # Genereer de scatter plot
    plt.scatter(x_vals, y_vals, s=1)  # s=1 om puntgrootte kleiner te maken

    # To generate inferno map
    #plt.scatter(sample_space)
    plt.colorbar()
    plt.title('Sample space')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.show()

    return


if __name__ == "__main__":

    #plot_samples()
    #timeing()
    importance_space(1000, 20)








# Approach 2:
"""
1. Look at different areas to see how they converge and what the difference is between them.

"""

