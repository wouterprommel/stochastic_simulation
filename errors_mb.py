import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
import Sample_methods
import time

#plt.rcParams['text.usetex'] = True

def eval_point_mandelbrot(x, y, i):
    max_iteration = i
    c = complex(x, y)
    z = 0
    iter = 0
    bounded = True
    itterating = True
    while bounded and itterating:
        bounded = abs(z) <= 2
        itterating = iter < max_iteration
        z = z*z + c
        iter += 1
    return (iter-1)/max_iteration # set red color (0-1)

def convergence(i, method='uniform', std=0.05):
    '''
    Compute |A_js - A_is| for all j < i
    return: list
    '''
    A_list =[]
    std_list = []
    #A_iN, std = mandelbrot.mc_area(N, i, std=True)
    A_is, std, n = gen_std_2(i, std, method=method)
    for j in range(10, i, 5):
        print('depth: ', j)

        #A_jN, std = mandelbrot.mc_area(N, j, std=True, method=method)
        Ajs, std, n = gen_std_2(samples, j)
        A_list.append(Ajs)
        std_list.append(std)
        print(n)

    A_j_list = np.array(A_list)
    deviation = np.abs(A_j_list - A_is)
    return deviation, std_list
def timeit():
    print(np.round(time.time() - t, 3), 'sec elapsed for random mb')

def convergence2(N, i, method='uniform'):
    t = time.time()
    N = int(N)

    if method == 'uniform':
        X = -1.5 + 3*np.random.rand(N)
        Y = -2 + 3*np.random.rand(N)
        samples = list(zip(X, Y))

    elif method == 'hypercube':
        samples = Sample_methods.hypercube(N)

    elif method == 'orthogonal':
        samples = Sample_methods.orthogonal(N)

    elif method == 'masking':
        img_size = 25
        i_space = 8
        Z_boundary = 1
        area_total, _, samples = Masking.masking(img_size, i_space, Z_boundary, N)
        N = len(samples)
    else:
        print('error no method')
        quit()

    samples = np.array(samples)
    print(len(samples))
    print(np.round(time.time() - t, 3), 'sec elapsed for sampling')
    t = time.time()

    A_list =[]
    std_list = []
    A_is, std = mc_samples(samples, i) 
    print(np.round(time.time() - t, 3), 'sec elapsed for A_is')
    t = time.time()
    for j in range(10, i, 10):
        print('depth: ', j)

        Ajs, std = mc_samples(samples, j)
        A_list.append(Ajs)
        std_list.append(std)
        print(np.round(time.time() - t, 3), 'sec elapsed for A_js')
        t = time.time()

    A_j_list = np.array(A_list)
    deviation = np.abs(A_j_list - A_is)
    return deviation, std_list

def mc_samples(samples, i):
    area_total = 9
    evaluations = []
    for x, y in samples:
        eval = eval_point_mandelbrot(x, y, i) == 1
        evaluations.append(eval)

    area = area_total * sum(evaluations) / len(evaluations)
    #print(f"Area from MC: {area=}")
    std_value = area_total * np.std(evaluations, ddof=1)/np.sqrt(len(evaluations)) # sample variance
    return area, std_value


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

def gen_std_2(i, std, method='uniform'):
    area_total = 9
    if method == 'uniform':
        N = int(1e8)
        n = int(np.sqrt(N))
        X = -1.5 + 3*np.random.rand(n)
        Y = -2 + 3*np.random.rand(n)
        samples = zip(X, Y)

    elif method == 'hypercube':
        N = int(1e6)
        samples = Sample_methods.hypercube(N)

    elif method == 'orthogonal':
        N = int(400*400)
        samples = Sample_methods.orthogonal(N)
    else:
        print('error')
        quit()

    samples = list(samples)

    evaluations = []

    std_value = 1
    n_samples = len(evaluations)
    while std_value > std and n_samples < N:
        for x, y in samples[n_samples:(n_samples + 1000)]:
            eval = eval_point_mandelbrot(x, y, i) == 1
            evaluations.append(eval)
        area = area_total * np.mean(evaluations) 
        std_value = area_total * np.std(evaluations, ddof=1)/np.sqrt(len(evaluations)) # sample variance
        #print(f'# samples: {len(evaluations)}, Area: {area}, Std: {std_value}')
        n_samples = len(evaluations)
    return area, std, n_samples



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

std = 0.1

plt.figure(figsize=(12, 8))
colors = ['tab:blue', 'tab:green', 'tab:red', "tab:orange"]
depth = 250
deviation2, std2 = convergence2(1e6, depth)
deviation1, std1 = convergence2(1e6, depth, method='orthogonal')
deviation3, std3 = convergence2(1e6, depth, method='hypercube')

plt.plot(list(range(10, depth, 10)), deviation1, label='$A_{j,s} - A_{i,s}$ ($10^{4}$ orthogonal samples)', color=colors[0], linestyle='-', alpha = 0.6)
plt.plot(list(range(10, depth, 10)), std1, label='$\sigma_{A_{i,s}}$ ($10^{4}$ orthogonal samples)', color=colors[0], linestyle='--')

plt.plot(list(range(10, depth, 10)), deviation2, label='$A_{j,s} - A_{i,s}$ ($10^{6}$ unifrom samples)', color=colors[2], linestyle='-', alpha = 0.6)
plt.plot(list(range(10, depth, 10)), std2, label='$\sigma_{A_{i,s}}$ ($10^{8}$ uniform samples)', color=colors[2], linestyle='--')

plt.plot(list(range(10, depth, 10)), deviation2, label='$A_{j,s} - A_{i,s}$ ($10^{8}$ hypercube samples)', color=colors[2], linestyle='-', alpha = 0.6)
plt.plot(list(range(10, depth, 10)), std2, label='$\sigma_{A_{i,s}}$ ($10^{8}$ hypercube samples)', color=colors[2], linestyle='--')
#plt.yscale('log')

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.legend(fontsize=20)
plt.title('Convergence of the area estimate', fontsize=30)
plt.xlabel('Iterations', fontsize=28)
plt.ylabel('$A_{j,s} - A_{i,s}$', fontsize=28)
plt.grid()
#plt.savefig(f'Figures/Convergence of area estimate.pdf', format='pdf')
plt.show()

# X, S, n = gen_std(10000, 80)
# print(f'{X} +- {1.96*S/np.sqrt(n)}')
