import Sample_methods
import importance
import numpy as np
import matplotlib.pyplot as plt
import time


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


def mc_area(N, i, method='uniform', std=False):
    # y from -1.5 to 1.5
    # x form -2 to 1
    n = int(np.sqrt(N))

    if method == 'uniform':
        X = -1.5 + 3*np.random.rand(n)
        Y = -2 + 3*np.random.rand(n)
        samples = zip(X, Y)
        area_total = 9

    elif method == 'hypercube':
        samples = Sample_methods.hypercube(N)
        area_total = 9

    elif method == 'orthogonal':
        samples = Sample_methods.orthogonal(N)
        area_total = 9

    elif method == 'importance':
        img_size = 100
        i_space = 5
        Z_boundary = 0.95
        area_total, samples = importance.importance(img_size, i_space, Z_boundary, N)

    evaluations = []
    for x, y in samples:
        eval = eval_point_mandelbrot(x, y, i) == 1
        evaluations.append(eval)

    area = area_total * sum(evaluations) / len(evaluations)
    #print(f"Area from MC: {area=}")
    std_value = area_total * np.std(evaluations, ddof=1)/np.sqrt(len(evaluations)) # sample variance

    if std == True:
        return area, std_value
    else: 
        return area

def pixel_count_area(img_size = 1000):

    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    image = np.zeros((img_size, img_size, 3)) # x, y, rgb
    max_iteration = 80

    # To generate inferno map
    mandelbrot_set = np.zeros((img_size, img_size))

    for j, x in enumerate(x_axis):
        if j % 100 == 0: print(j)
        for i, y in enumerate(y_axis):
            color = eval_point_mandelbrot(x, y, max_iteration)

            # To generate inferno map
            mandelbrot_set[i, j] = eval_point_mandelbrot(x, y, max_iteration)
            image[i, j, 0] = color # set red color (0-1)

    S = np.sum(image[:, :, 0] == 1)
    A = S/(img_size*img_size) * 3*3
    print(f"Area from pixel count: {A=}")

    plt.imshow(image)
    plt.show()

    # To generate inferno map
    plt.imshow(mandelbrot_set, extent=(-2, 1, -1.5, 1.5), cmap='inferno')
    plt.colorbar()
    plt.title('Mandelbrot Fractal')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
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
    print(mc_area(N, 80, 'importance'))
    print (np.round(time.time() - t, 3), 'sec elapsed for importance mb')

def plot_samples():
    ''' 
    plot sample points
    '''
    N = int(25)
    samples = Sample_methods.hypercube(N)
    #samples2 = hypercube.hypercube(N) # set func to return samples first
    samples2 = Sample_methods.orthogonal(N) # set func to return samples first
    plt.scatter(*zip(*samples))
    plt.scatter(*zip(*samples2))
    plt.show()

if __name__ == "__main__":

    #plot_samples()
    #pixel_count_area()
    timeing()

    # single value itteration
    """ a = 0.28
    b = 0.03
    xn = 0
    yn = 0
    X = [xn]
    Y = [yn]
    for i in range(100):
        xnp1 = xn*xn - yn*yn + a # x_{n+1}
        ynp1 = 2*xn*yn + b # y_{n+1}
        X.append(xnp1)
        Y.append(ynp1)
        xn = xnp1
        yn = ynp1

    print(X)
    print(Y)

    plt.scatter(X, Y)
    plt.show() """