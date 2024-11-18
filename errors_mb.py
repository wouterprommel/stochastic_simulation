'''
This module contains functions for computing 
the error of the area estimate of the Mandelbrot set.
'''
import matplotlib.pyplot as plt
import numpy as np
import mandelbrot

def convergence(N, i, method='uniform'):
    '''
    Compute |A_js - A_is| for all j < i
    return: list
    '''
    A_list =[]
    std_list = []
    A_iN, std = mandelbrot.mc_area(N, i, std=True, method=method)
    for j in range(10, i):
        A_jN, std = mandelbrot.mc_area(N, j, std=True, method=method)
        std_list.append(std)
        A_list.append(A_jN)

    A_j_list = np.array(A_list)
    deviation = np.abs(A_j_list - A_iN)
    return deviation, std_list

def result(N, i, method='uniform'):
    '''
    Compute Area 10x give confidence interval at p=95%
    '''
    area_list = []
    for n in range(10):
        area_list.append(mandelbrot.mc_area(N, i, method=method))

    S = np.std(area_list, ddof=1)
    n = len(area_list)
    a = 1.96*S/np.sqrt(n) # Radius of confidence interval

    return f'{method}: {np.mean(np.array(area_list))} +- {a}, used {n} simulations'

print(result(int(400*400), 150, 'uniform'))
print(result(int(400*400), 150, 'hypercube'))
print(result(int(400*400), 150, 'orthogonal'))
print(result(int(400*400), 150, 'adaptive'))

deviation1, std1 = convergence(1e4, 200)
deviation2, std2 = convergence(1e4, 200, method='orthogonal')
deviation3, std3 = convergence(1e4, 200, method='hypercube')
deviation4, std4 = convergence(1e4, 200, method='adaptive')

colors = ['tab:blue', 'tab:green', 'tab:red', "tab:orange"]

plt.figure(figsize=(5.91, 3.6))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(deviation1, label='$|A_{j,s} - A_{i,s}|$ (Uniform sampling)',
         color=colors[0],  linestyle='-', alpha = 0.6)
plt.plot(deviation3, label='$|A_{j,s} - A_{i,s}|$ (Hypercube sampling)',
         color=colors[1],  linestyle='-', alpha = 0.6)
plt.plot(deviation2, label='$|A_{j,s} - A_{i,s}|$ (Orthogonal sampling)',
         color=colors[2], linestyle='-', alpha = 0.6)
plt.plot(deviation4, label='$|A_{j,s} - A_{i,s}|$ (Custom sampling)',
         color=colors[3],  linestyle='-', alpha = 0.6)
plt.tick_params(axis='x', labelsize=9)
plt.tick_params(axis='y', labelsize=9)
plt.legend(fontsize=9)
plt.xlabel('Iterations', fontsize=9)
plt.ylabel('$|A_{j,s} - A_{i,s}|$', fontsize=9)
plt.grid()
plt.savefig('Figures/Convergence_n=1e4_i=200.pdf', format='pdf')
plt.show()
