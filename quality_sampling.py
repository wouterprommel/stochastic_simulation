import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
import Sample_methods as sm

sample_sizes = [10**2, 20**2, 30**2, 40**2, 50**2, 60**2, 70**2, 80**2, 90**2, 100**2, 200**2, 300**2, 400**2, 500**2, 600**2, 700**2, 800**2, 900**2, 1000**2] #, 5000**2]	#10e8

def plot_comparison(sample_sizes, max_iter):
    lit_area = 1.5052 # literature value
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
    plt.axhline(y = lit_area, color = colors[3], linestyle = '--', label = "Literature Area") 
    plt.xlabel('Number of Samples', fontsize=14)
    plt.ylabel('Estimated Area', fontsize=14)
    plt.title(f'Estimated Area vs Number of Samples for {max_iter} iterations', fontsize=16)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()
    #plt.savefig(f'Estimated_Area_vs_Number_of_Samples_{max_iter}_iterations.pdf', format='pdf')
    plt.show()
    return uniform_areas, uniform_errors, hypercube_areas, ortho_areas, ortho_errors

plot_comparison(sample_sizes, 50)
plot_comparison(sample_sizes, 100)
plot_comparison(sample_sizes, 200)
plot_comparison(sample_sizes, 400)

def error_evaluation(area, lit_area):
    mean_area = np.mean(area)
    std_area = np.std(area, ddof=1)
    print(f"""The mean of the estimated areas is {mean_area}""")	
    abs_error = np.abs(mean_area - lit_area) # just an estimate of the average absolute error
    rel_error = abs_error / lit_area * 100 # in percentage
   # print(f"""The absolute error is {abs_error}""")
   # print(f"""The relative error is {rel_error}""")
    return abs_error, rel_error, mean_area, std_area
print("Uniform sampling:")
#error_evaluation(uniform_areas, lit_area)
print("Hypercube sampling:")
#error_evaluation(hypercube_areas, lit_area)
print("Orthogonal sampling:")
#error_evaluation(ortho_areas, lit_area)

# 1 plot log en 1 zonder log 