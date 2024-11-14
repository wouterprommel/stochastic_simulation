import mandelbrot
import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams['text.usetex'] = True

def convergence(N, i):
    '''
    Compute |A_js - A_is| for all j < i
    return: list
    '''
    A_list =[]
    A_iN = mandelbrot.mc_area(N, i)
    for j in range(10, i):
        if j % i//2 == 0: print('half')

        A_jN = mandelbrot.mc_area(N, j)
        A_list.append(A_jN)

    A_j_list = np.array(A_list)
    deviation = np.abs(A_j_list - A_iN)
    return deviation


def std_mandelbrot(N, i, M=10):
    '''
    std of M Areas
    '''
    list = []
    for n in range(M):
        list.append(mandelbrot.mc_area(N, i))
    std = np.std(np.array(list))
    return std

def sample_std_mandelbrot(N, i, M=10):
    '''
    sample variance -- std. so n-1. zelfde als ddof=1 in np.std.
    '''
    Xi = []
    for n in range(M):
        Xi.append(mandelbrot.mc_area(N, i))
    Xi = np.array(Xi)
    Xmean = Xi.mean()
    sample_var = np.sum((Xi - Xmean)**2)/(len(Xi) -1)

    return np.sqrt(sample_var)

def sample_var_from_list(list):
    '''
    input: list of area sims
    output: sample variance
    '''
    Xi = np.array(list)
    Xmean = Xi.mean()
    sample_var = np.sum((Xi - Xmean)**2)/(len(Xi) - 1)
    return sample_var

def gen_std(N, i, Set_std=0.0015):
    '''
    input A_iN (Xi), wanted std
    output, mean(X), sample std S, n
    '''
    list = []
    for n in range(3):
        list.append(mandelbrot.mc_area(N, i))

    S2 = sample_var_from_list(list)
    n = len(list)
    while np.sqrt(S2/n) > Set_std:
        list.append(mandelbrot.mc_area(N, i))
        S2 = sample_var_from_list(list)
        n = len(list)
        if n % 100 == 0:
            print(n, np.sqrt(S2/n))


    return np.mean(np.array(list)), np.sqrt(S2), n


deviation_n3 = convergence(1e7, 120)
#deviation_n4 = convergence(1e8, 120)
print(deviation_n3)
plt.plot(deviation_n3)
#plt.plot(deviation_n4)
plt.xlabel('iterations')
plt.ylabel('$A_{j,s} - A_{i,s}$')
plt.show()

# X, S, n = gen_std(10000, 80)
# print(f'{X} +- {1.96*S/np.sqrt(n)}')