'''
This module computes the Mandelbrot set and estimates its area using a pixel counting method and Monte Carlo methods.
'''
import time
import numpy as np
import matplotlib.pyplot as plt


import Sample_methods
import Masking

def eval_point_mandelbrot(x, y, i):
    '''
    Determine if a point is in the mandelbrot set
    return: range 0-1
    '''
    max_iteration = i
    c = complex(x, y)
    z = 0
    iter = 0
    bounded = True
    iterating = True
    while bounded and iterating:
        bounded = abs(z) <= 2
        iterating = iter < max_iteration
        z = z*z + c
        iter += 1
    return (iter-1)/max_iteration # set red color (0-1)

def mc_area(N, i, method='uniform', std=False):
    '''
    Compute the area of the Mandelbrot set using different sampling methods
    return: float
    '''
    area_total = 9
    N = int(N)
    if method == 'uniform':
        X = -1.5 + 3*np.random.rand(N)
        Y = -2 + 3*np.random.rand(N)
        samples = list(zip(X, Y))

    elif method == 'hypercube':
        samples = Sample_methods.hypercube(N)

    elif method == 'orthogonal':
        samples = Sample_methods.orthogonal(N)

    elif method == 'adaptive':
        img_size = 25
        i_space = 8
        Z_boundary = 1
        area_total, _, samples = Masking.adaptive(img_size, i_space, Z_boundary, N)
        N = len(samples)

    evaluations = []
    for x, y in samples:
        eval = eval_point_mandelbrot(x, y, i) == 1
        evaluations.append(eval)

    area = area_total * sum(evaluations) / len(evaluations)
    std_value = area_total * np.std(evaluations, ddof=1)/np.sqrt(len(evaluations)) # sample variance

    if std is True:
        return area, std_value
    else:
        return area


def pixel_count_area(img_size = 1000):
    '''	
    Computes the area of the Mandelbrot set using a pixel counting method
    return: float
    '''
    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    max_iteration = 80

    # To generate inferno map
    mandelbrot_set = np.zeros((img_size, img_size))

    for j, x in enumerate(x_axis):
        for i, y in enumerate(y_axis):
            mandelbrot_set[i, j] = eval_point_mandelbrot(x, y, max_iteration)

    S = np.sum(mandelbrot_set[:, :] == 1)
    A = S/(img_size*img_size) * 3*3
    print(f"Area from pixel count: {A=}")
    print(f"The relative error compared to the literature value is: {np.abs(A - 1.5065)/1.5065*100}%.")

    plt.figure(figsize=(5.91/2, 3.6/2))
    plt.imshow(mandelbrot_set, extent=(-2, 1, -1.5, 1.5), cmap='inferno')
    colorbar = plt.colorbar()
    colorbar.ax.tick_params(labelsize=8)
    plt.xlabel('Real Part', fontsize=8)
    plt.ylabel('Imaginary Part', fontsize=8)
    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.savefig(f'Figures/Mandelbrot.pdf', bbox_inches='tight', format='pdf')
    plt.show()

def timeing():
    '''
    Time to sample N amount of points for different sampling methods
    '''
    N = int(1e4)

    t = time.time()
    print(mc_area(N, 80))
    print (np.round(time.time() - t, 3), 'sec elapsed for random mb')

    t = time.time()
    print(mc_area(N, 80, 'hypercube'))
    print (np.round(time.time() - t, 3), 'sec elapsed for hypercube mb')

    t = time.time()
    print(mc_area(N, 80, 'orthogonal'))
    print (np.round(time.time() - t, 3), 'sec elapsed for orthogonal mb')

    t = time.time()
    print(mc_area(N, 80, 'adaptive'))
    print (np.round(time.time() - t, 3), 'sec elapsed for adaptive orthogonal mb')

def plot_samples():
    ''' 
    Plot sample points
    '''
    N = int(25)
    samples = Sample_methods.hypercube(N)
    samples2 = Sample_methods.orthogonal(N) # Set func to return samples first
    plt.scatter(*zip(*samples))
    plt.scatter(*zip(*samples2))
    plt.show()


if __name__ == "__main__":
    # Use LaTex font for labels
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    pixel_count_area()
    timeing()
