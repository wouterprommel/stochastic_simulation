import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
import Sample_methods as sm

sample_sizes = [80**2, 90**2, 100**2, 110**2, 120**2, 130**2, 140**2, 150**2, 160**2, 170**2, 180**2, 190**2, 200**2]	
max_iter = 80
pixel_area = A=1.5536880000000002
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
colors = ['tab:blue', 'tab:green', 'tab:red', "tab:purple"]
markers = ['o', 's', '^']

plt.errorbar(sample_sizes, uniform_areas, yerr=uniform_errors, label='Uniform sampling', color=colors[0], marker='o', linestyle='-', capsize=5)
plt.errorbar(sample_sizes, hypercube_areas, yerr=hypercube_errors, label='Hypercube sampling', color=colors[1], marker='s', linestyle='-', capsize=5)
plt.errorbar(sample_sizes, ortho_areas, yerr=ortho_errors, label='Orthogonal sampling', color=colors[2], marker='^', linestyle='-', capsize=5)
plt.axhline(y = pixel_area, color = colors[3], linestyle = '--', label = "Pixel Area") 
plt.xlabel('Number of Samples', fontsize=14)
plt.ylabel('Estimated Area', fontsize=14)
plt.title('Estimated Area vs Number of Samples', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

def error_evaluation(area, pixel_area):
    mean_area = np.mean(area)
    std_area = np.std(area, ddof=1)
    print(f"""The mean of the estimated areas is {mean_area}""")	
    abs_error = np.abs(mean_area - pixel_area) # just an estimate of the average absolute error
    rel_error = abs_error / pixel_area * 100 # in percentage
    print(f"""The absolute error is {abs_error}""")
    print(f"""The relative error is {rel_error}""")
    return abs_error, rel_error, mean_area, std_area
print("Uniform sampling:")
error_evaluation(uniform_areas, pixel_area)
print("Hypercube sampling:")
error_evaluation(hypercube_areas, pixel_area)
print("Orthogonal sampling:")
error_evaluation(ortho_areas, pixel_area)

