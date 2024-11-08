import numpy as np
import matplotlib.pyplot as plt


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

def mc_area(N, i):
    # y from -1.5 to 1.5
    # x form -2 to 1
    X = -1.5 + 3*np.random.rand(N)
    Y = -2 + 3*np.random.rand(N)
    evaluations = []
    for x, y in zip(X, Y):
        eval = eval_point_mandelbrot(x, y, i) == 1
        evaluations.append(eval)

    area = 9 * sum(evaluations) / len(evaluations)
    #print(f"Area from MC: {area=}")

    return area

def pixel_count_area(img_size = 1000):

    x_axis = np.linspace(-2, 1, img_size)
    y_axis = np.linspace(-1.5, 1.5, img_size)
    image = np.zeros((img_size, img_size, 3)) # x, y, rgb
    max_iteration = 80

    for j, x in enumerate(x_axis):
        if j % 100 == 0: print(j)
        for i, y in enumerate(y_axis):
            color = eval_point_mandelbrot(x, y, 50)
            image[i, j, 0] = color # set red color (0-1)

    S = np.sum(image[:, :, 0] == 1)
    A = S/(img_size*img_size) * 3*3
    print(f"Area from pixel count: {A=}")

    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    #mc_area(1000000)
    #pixel_count_area()

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