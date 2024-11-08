import mandelbrot
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

def convergence():
    i = 150
    N = 100000
    A_list =[]
    error_i_list = []
    error_N_list = []
    A_iN = mandelbrot.mc_area(N, i)
    for j in range(10, i, 10):
        if j % 50 == 0: print('half')

        A_jN = mandelbrot.mc_area(N, j)
        er_i = std_mandelbrot(N, j)
        error_i_list.append(er_i)
        A_list.append(A_jN)

    A_j_list = np.array(A_list)
    deviation = A_j_list - A_iN

    plt.plot(deviation)
    plt.plot(error_i_list)
    plt.xlabel('iterations')
    plt.ylabel('$$A_{j,s} - A_{i,s}$$')
    plt.show()

def std_mandelbrot(N, i, M=10):
    list = []
    for n in range(M):
        list.append(mandelbrot.mc_area(N, i))
    std = np.std(np.array(list))
    return std

def sample_std_mandelbrot(N, i, M=10):
    Xi = []
    for n in range(M):
        Xi.append(mandelbrot.mc_area(N, i))
    Xi = np.array(Xi)
    Xmean = Xi.mean()
    sample_var = np.sum((Xi - Xmean)**2)/(len(Xi) -1)

    return np.sqrt(sample_var)

def sample_var_from_list(list):
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


X, S, n = gen_std(10000, 30)
print(f'{X} +- {1.96*S/np.sqrt(n)}')