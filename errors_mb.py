"""
Error Computation of Mandelbrot Set Area Estimates

This module contains functions for computing the error of the area 
estimate of the Mandelbrot set. It includes methods for analyzing 
convergence and computing intervals of the area estimates.

This module includes:

1. Convergence Analysis
    - Computes the deviation of area estimates over increasing 
      iterations.

2. Confidence Interval Estimation
    - Computes the mean area estimate with a 95% confidence interval.

Usage:
The module can be run directly to compute area estimates for different 
sampling methods and plot convergence of area estimates.
"""

import numpy as np
import matplotlib.pyplot as plt

import mandelbrot


def convergence(N, i, method='uniform'):
    """
    Compute the convergence of the Mandelbrot set area estimate by 
    calculating the deviation |A_js - A_is| for all j < i.

    Parameters:
    N (int): Number of samples used for area estimation.
    i (int): Maximum number of iterations for Mandelbrot evaluation.
    method (str): Sampling method ('uniform', 'hypercube', 
        'orthogonal', 'adaptive'). Default is 'uniform'.

    Returns:
    deviation (list): List of absolute deviations |A_js - A_is|.
    std_list (list): List of standard deviations for area estimates.
    """
    A_list = []
    std_list = []

    # Compute area estimate for final iteration.
    A_iN, std = mandelbrot.mc_area(N, i, std=True, method=method)

    # Compute area estimates for all iterations j < i.
    for j in range(10, i):
        A_jN, std = mandelbrot.mc_area(N, j, std=True, method=method)
        std_list.append(std)
        A_list.append(A_jN)

    # Compute absolute deviations.
    A_j_list = np.array(A_list)
    deviation = np.abs(A_j_list - A_iN)

    return deviation, std_list


def result(N, i, method='uniform'):
    """
    Compute the area of the Mandelbrot set 10 times and calculate a 
    95% confidence interval for the mean area estimate.

    Parameters:
    N (int): Number of samples used for area estimation.
    i (int): Maximum number of iterations for Mandelbrot evaluation.
    method (str): Sampling method ('uniform', 'hypercube', 
        'orthogonal', 'adaptive'). Default is 'uniform'.

    Returns: A formatted string containing the mean area estimate, 
    confidence interval, and the number of simulations used.
    """
    area_list = []

    # Compute area estimates.
    for n in range(10):
        area_list.append(mandelbrot.mc_area(N, i, method=method))

    # Compute standard deviation and confidence interval.
    S = np.std(area_list, ddof=1)
    n = len(area_list)
    a = 1.96*S/np.sqrt(n)  # Radius of confidence interval

    return (
    f"{method}: {np.mean(np.array(area_list))} +- {a}, "
    f"used {n} simulations"
    )


if __name__ == "__main__":
    # Compute and display results for different sampling methods.
    print(result(int(400*400), 150, 'uniform'))
    print(result(int(400*400), 150, 'hypercube'))
    print(result(int(400*400), 150, 'orthogonal'))
    print(result(int(400*400), 150, 'adaptive'))

    # Analyze convergence for different sampling methods.
    deviation1, std1 = convergence(1e4, 200)
    deviation2, std2 = convergence(1e4, 200, method='orthogonal')
    deviation3, std3 = convergence(1e4, 200, method='hypercube')
    deviation4, std4 = convergence(1e4, 200, method='adaptive')

    # Plot convergence analysis.
    colors = ['tab:blue', 'tab:green', 'tab:red', "tab:orange"]

    plt.figure(figsize=(5.91, 3.6))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot deviations for each method.
    plt.plot(deviation1, label='$|A_{j,s} - A_{i,s}|$ (Uniform sampling)',
             color=colors[0],  linestyle='-', alpha=0.6)
    plt.plot(deviation3, label='$|A_{j,s} - A_{i,s}|$ (Hypercube sampling)',
             color=colors[1],  linestyle='-', alpha=0.6)
    plt.plot(deviation2, label='$|A_{j,s} - A_{i,s}|$ (Orthogonal sampling)',
             color=colors[2], linestyle='-', alpha=0.6)
    plt.plot(deviation4, label='$|A_{j,s} - A_{i,s}|$ (Custom sampling)',
             color=colors[3],  linestyle='-', alpha=0.6)
    plt.tick_params(axis='x', labelsize=9)
    plt.tick_params(axis='y', labelsize=9)
    plt.legend(fontsize=9)
    plt.xlabel('Iterations', fontsize=9)
    plt.ylabel('$|A_{j,s} - A_{i,s}|$', fontsize=9)
    plt.grid()
    plt.savefig('Figures/Convergence_n=1e4_i=200.pdf', format='pdf')
    plt.show()
