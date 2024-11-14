import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
import Sample_methods as sm

sample_sizes = [900, 4900, 10000]	
max_iters = [50, 100, 150]

for s in sample_sizes:
    print(f"""sample sieis {s}""")
    for n in max_iters:
        print(f"""max iters {n}""")
        uniform = mandelbrot.mc_area(s, n, method='uniform', std=True)
        print(uniform)
        hypercube = mandelbrot.mc_area(s, n, method='hypercube', std=True)
        print(hypercube)
        ortho  = mandelbrot.mc_area(s, n, method='orthogonal', std=True)
        print(ortho)