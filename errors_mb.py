import mandelbrot
import matplotlib.pyplot as plt
import numpy as np

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

def result(N, i, method='uniform'):
    '''
    Compute Area 10x give confidence interval at p=95%
    '''
    list = []
    for n in range(10):
        list.append(mandelbrot.mc_area(N, i, method=method))

    S = np.std(list, ddof=1)
    n = len(list)
    a = 1.96*S/np.sqrt(n) # a radius of confidence interval

    return f'{method}: {np.mean(np.array(list))} +- {a}, used {n} simulations'


print(result(int(400*400), 150, 'uniform'))
print(result(int(400*400), 150, 'hypercube'))
print(result(int(400*400), 150, 'orthogonal'))
print(result(int(400*400), 150, 'masking'))

quit()

plt.figure(figsize=(12, 8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(5.91, 3.6))
colors = ['tab:blue', 'tab:green', 'tab:red', "tab:orange"]
deviation1, std1 = convergence(1e4, 200)
deviation2, std2 = convergence(1e4, 200, method='orthogonal')
deviation3, std3 = convergence(1e4, 200, method='hypercube')
deviation4, std4 = convergence(1e4, 200, method='masking')

plt.plot(deviation1, label='$|A_{j,s} - A_{i,s}|$ (Uniform sampling)', color=colors[0],  linestyle='-', alpha = 0.6)
#plt.plot(std1, label='$\sigma_{A_{i,s}}$ ($10^{4}$ samples)', color=colors[0], linestyle='--')

plt.plot(deviation3, label='$|A_{j,s} - A_{i,s}|$ (Hypercube sampling)', color=colors[1],  linestyle='-', alpha = 0.6)
plt.plot(deviation2, label='$|A_{j,s} - A_{i,s}|$ (Orthogonal sampling)', color=colors[2], linestyle='-', alpha = 0.6)
plt.plot(deviation4, label='$|A_{j,s} - A_{i,s}|$ (Custom sampling)', color=colors[3],  linestyle='-', alpha = 0.6)
#plt.plot(std2, label='$\sigma_{A_{i,s}}$ ($10^{8}$ samples)', color=colors[2], linestyle='--')
#plt.yscale('log')
plt.tick_params(axis='x', labelsize=9)
plt.tick_params(axis='y', labelsize=9)
plt.legend(fontsize=9)
#plt.title('Convergence of the area estimate', fontsize=30)
plt.xlabel('Iterations', fontsize=9)
plt.ylabel('$|A_{j,s} - A_{i,s}|$', fontsize=9)
plt.grid()
plt.savefig(f'Figures/Convergence_n=1e4_i=200.pdf', format='pdf')
plt.show()

# X, S, n = gen_std(10000, 80)
# print(f'{X} +- {1.96*S/np.sqrt(n)}')
