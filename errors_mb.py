import mandelbrot
import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams['text.usetex'] = True

def convergence(N, i, method='uniform'):
    '''
    Compute |A_js - A_is| for all j < i
    return: list
    '''
    A_list =[]
    std_list = []
    A_iN, std = mandelbrot.mc_area(N, i, std=True, method=method)
    for j in range(10, i):
        if j % (i//2) == 0: print('half')

        A_jN, std = mandelbrot.mc_area(N, j, std=True, method=method)
        std_list.append(std)
        A_list.append(A_jN)

    A_j_list = np.array(A_list)
    deviation = np.abs(A_j_list - A_iN)
    return deviation, std_list


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

plt.figure(figsize=(12, 8))
colors = ['tab:blue', 'tab:green', 'tab:red', "tab:orange"]
deviation1, std1 = convergence(1e4, 200)
deviation2, std2 = convergence(1e4, 200, method='orthogonal')
deviation3, std3 = convergence(1e4, 200, method='hypercube')
deviation4, std4 = convergence(1e4, 200, method='masking')

plt.plot(deviation1, label='$|A_{j,s} - A_{i,s}|$ (method: uniform)', color=colors[0],  linestyle='-', alpha = 0.6)
#plt.plot(std1, label='$\sigma_{A_{i,s}}$ ($10^{4}$ samples)', color=colors[0], linestyle='--')

plt.plot(deviation3, label='$|A_{j,s} - A_{i,s}|$ (method: hypercube)', color=colors[1],  linestyle='-', alpha = 0.6)
plt.plot(deviation2, label='$|A_{j,s} - A_{i,s}|$ (method: orthogonal)', color=colors[2], linestyle='-', alpha = 0.6)
plt.plot(deviation4, label='$|A_{j,s} - A_{i,s}|$ (method: masking)', color=colors[3],  linestyle='-', alpha = 0.6)
#plt.plot(std2, label='$\sigma_{A_{i,s}}$ ($10^{8}$ samples)', color=colors[2], linestyle='--')
#plt.yscale('log')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(fontsize=20)
#plt.title('Convergence of the area estimate', fontsize=30)
plt.xlabel('Iterations', fontsize=28)
plt.ylabel('$|A_{j,s} - A_{i,s}|$', fontsize=28)
plt.grid()
plt.savefig(f'Figures/Convergence_n=1e4_i=200.pdf', format='pdf')
plt.show()

# X, S, n = gen_std(10000, 80)
# print(f'{X} +- {1.96*S/np.sqrt(n)}')
