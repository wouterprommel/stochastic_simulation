import mandelbrot
#import errors_mb

import matplotlib.pyplot as plt
import numpy as np

# Rooks in chess

def hypercube(N):
    """
    Applies Latin hypercube sampling to the mandelbroth set grid of x and y values 
    with supplied sample size N.
    """
    # Configure Chess board lists from dimensions
    x_axis = np.linspace(-2, 1, N).tolist()
    y_axis = np.linspace(-1.5, 1.5, N).tolist()

    samples = []

    # Assigns x and y value to samples from list of options and removes options
    for _ in range(N):
        x = np.random.choice(x_axis)
        y = np.random.choice(y_axis)

        samples.append([x, y])

        x_axis.remove(x)
        y_axis.remove(y)


    # Maybe make 1 function for this (this is done in many functions)
    # Supply samples to evaluate mandelbroth
    max_iteration = 50
    evaluations = []

    # Generates list of mandelbroth evaluations, that is returned (for now)
    for x, y in samples:
        evaluation = mandelbrot.eval_point_mandelbrot(x, y, max_iteration)
        evaluations.append(evaluation)
        print(f"Evaluated point ({x}, {y}): {evaluation}")

    return evaluations

N = 5
hypercube(N)