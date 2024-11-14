import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
import Sample_methods as sm

sample_sizes = [80**2, 90**2, 100**2, 110**2, 120**2, 130**2, 140**2, 150**2, 160**2, 170**2, 180**2, 190**2, 200**2]	
max_iter = 80

uniform_areas = []
uniform_errors = []
hypercube_areas = []
hypercube_errors = []
ortho_areas = []
ortho_errors = []

# Loop over sample sizes and collect results
for s in sample_sizes:
    print(f"Sample size: {s}")
    uniform_A, uniform_err = mandelbrot.mc_area(s, max_iter, method='uniform', std=True)
    hypercube_A, hypercube_err = mandelbrot.mc_area(s, max_iter, method='hypercube', std=True)
    ortho_A, ortho_err = mandelbrot.mc_area(s, max_iter, method='orthogonal', std=True)  
    
    uniform_areas.append(uniform_A)
    uniform_errors.append(uniform_err)
    hypercube_areas.append(hypercube_A)
    hypercube_errors.append(hypercube_err)
    ortho_areas.append(ortho_A)
    ortho_errors.append(ortho_err)

plt.figure(figsize=(12, 8))
method_names = ['Uniform', 'Hypercube', 'Orthogonal']
colors = ['red', 'green', 'blue']
markers = ['o', 's', '^']

plt.errorbar(sample_sizes, uniform_areas, yerr=uniform_errors, label='Uniform sampling', color=colors[0], marker='o', linestyle='-', capsize=5)
plt.errorbar(sample_sizes, hypercube_areas, yerr=hypercube_errors, label='Hypercube sampling', color=colors[1], marker='s', linestyle='-', capsize=5)
plt.errorbar(sample_sizes, ortho_areas, yerr=ortho_errors, label='Orthogonal sampling', color=colors[2], marker='^', linestyle='-', capsize=5)

plt.xlabel('Number of Samples', fontsize=14)
plt.ylabel('Estimated Area', fontsize=14)
plt.title('Estimated Area vs Number of Samples', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

std_uniform = np.std(uniform_areas, ddof=1)
mean_uniform = np.mean(uniform_areas)
std_hypercube = np.std(hypercube_areas, ddof=1)
mean_hypercube = np.mean(hypercube_areas)
std_ortho = np.std(ortho_areas, ddof=1)
mean_ortho = np.mean(ortho_areas)
print(f"""The mean of the estimated areas for the uniform sampling is {mean_uniform}""")
print(f"""The standard deviation of the estimated areas for the uniform sampling is {std_uniform}""")
print(f"""The mean of the estimated areas for the hypercube sampling is {mean_hypercube}""")
print(f"""The standard deviation of the estimated areas for the hypercube sampling is {std_hypercube}""")
print(f"""The mean of the estimated areas for the orthogonal sampling is {mean_ortho}""")
print(f"""The standard deviation of the estimated areas for the orthogonal sampling is {std_ortho}""")
