import numpy as np
import matplotlib.pyplot as plt

img_size = 1000

x_axis = np.linspace(-2, 1, img_size)
y_axis = np.linspace(-1.4, 1.4, img_size)
image = np.zeros((img_size, img_size, 3)) # x, y, rgb
max_iteration = 80

for j, x in enumerate(x_axis):
    if j % 100 == 0: print(j)
    for i, y in enumerate(y_axis):
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
        image[i, j, 0] = (iter-1)/max_iteration # set red color (0-1)
        #print(image[i, j, 0])

plt.imshow(image)
plt.show()


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