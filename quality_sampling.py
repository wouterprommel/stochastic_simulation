import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
import Sample_methods as sm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

sample_sizes = [10**2, 20**2, 30**2, 40**2, 50**2, 60**2, 70**2, 80**2, 90**2, 100**2, 200**2, 300**2, 400**2, 500**2, 600**2, 700**2, 800**2, 900**2, 1000**2] #, 5000**2]	#10e8
lit_area = 1.5052 # literature value
"""
def plot_comparison(sample_sizes, max_iter):
    lit_area = 1.5052 # literature value
    uniform_areas = []
    uniform_errors = []
    hypercube_areas = []
    hypercube_errors = []
    ortho_areas = []
    ortho_errors = []
    masking_areas = []
    masking_errors = []
    
    # Loop over sample sizes and collect results
    for s in sample_sizes:
        print(f"Sample size: {s}")
        uniform_A, uniform_err = mandelbrot.mc_area(s, max_iter, method='uniform', std=True)
        hypercube_A, hypercube_err = mandelbrot.mc_area(s, max_iter, method='hypercube', std=True)
        ortho_A, ortho_err = mandelbrot.mc_area(s, max_iter, method='orthogonal', std=True)  
        masking_A, masking_err = mandelbrot.mc_area(s, max_iter, method='masking', std=True)
        
        uniform_areas.append(uniform_A)
        uniform_errors.append(uniform_err)
        hypercube_areas.append(hypercube_A)
        hypercube_errors.append(hypercube_err)
        ortho_areas.append(ortho_A)
        ortho_errors.append(ortho_err)
        masking_areas.append(masking_A)
        masking_errors.append(masking_err)

    plt.figure(figsize=(12, 8))
    method_names = ['Uniform', 'Hypercube', 'Orthogonal']
    colors = ['tab:blue', 'tab:green', 'tab:red', "tab:purple", "tab:orange"]
    markers = ['o', 's', '^']

    plt.errorbar(sample_sizes, uniform_areas, yerr=uniform_errors, label='Uniform sampling', color=colors[0], marker='o', linestyle='-', capsize=5)
    plt.errorbar(sample_sizes, hypercube_areas, yerr=hypercube_errors, label='Hypercube sampling', color=colors[1], marker='s', linestyle='-', capsize=5)
    plt.errorbar(sample_sizes, ortho_areas, yerr=ortho_errors, label='Orthogonal sampling', color=colors[2], marker='^', linestyle='-', capsize=5)
    plt.errorbar(sample_sizes, masking_areas, yerr=masking_errors, label='Masking sampling', color=colors[4], marker='x', linestyle='-', capsize=5)
    plt.axhline(y = lit_area, color = colors[3], linestyle = '--', label = "Literature Area") 
    plt.xlabel('Number of Samples', fontsize=18)
    plt.ylabel('Estimated Area', fontsize=18)
    plt.title(f'Estimated Area vs Number of Samples for {max_iter} iterations', fontsize=20)
    plt.legend(fontsize=16)
    plt.xscale('log')
    plt.grid(True)
    plt.tight_layout()

    #plt.savefig(f'Estimated_Area_vs_Number_of_Samples_{max_iter}_iterations.pdf', format='pdf')
    plt.show()
    
    return uniform_areas, uniform_errors, hypercube_areas, ortho_areas, ortho_errors
"""


def plot_comparison(sample_sizes, max_iter):
    uniform_areas, uniform_errors = [], []
    hypercube_areas, hypercube_errors = [], []
    ortho_areas, ortho_errors = [], []
    masking_areas, masking_errors = [], []

    # Calculate areas and errors for each method
    for s in sample_sizes:
        print(f"Sample size: {s}")
        uniform_A, uniform_err = mandelbrot.mc_area(s, max_iter, method='uniform', std=True)
        hypercube_A, hypercube_err = mandelbrot.mc_area(s, max_iter, method='hypercube', std=True)
        ortho_A, ortho_err = mandelbrot.mc_area(s, max_iter, method='orthogonal', std=True)
        masking_A, masking_err = mandelbrot.mc_area(s, max_iter, method='masking', std=True)

        uniform_areas.append(uniform_A)
        uniform_errors.append(uniform_err)
        hypercube_areas.append(hypercube_A)
        hypercube_errors.append(hypercube_err)
        ortho_areas.append(ortho_A)
        ortho_errors.append(ortho_err)
        masking_areas.append(masking_A)
        masking_errors.append(masking_err)

    # Main plot
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['tab:blue', 'tab:green', 'tab:red', "tab:purple", "tab:orange"]
    ax.errorbar(sample_sizes, uniform_areas, yerr=uniform_errors, label='Uniform sampling', color=colors[0], marker='o', linestyle='-', capsize=5)
    ax.errorbar(sample_sizes, hypercube_areas, yerr=hypercube_errors, label='Hypercube sampling', color=colors[1], marker='s', linestyle='-', capsize=5)
    ax.errorbar(sample_sizes, ortho_areas, yerr=ortho_errors, label='Orthogonal sampling', color=colors[2], marker='^', linestyle='-', capsize=5)
    ax.errorbar(sample_sizes, masking_areas, yerr=masking_errors, label='Masking sampling', color=colors[4], marker='x', linestyle='-', capsize=5)
    ax.axhline(y=lit_area, color=colors[3], linestyle='--', label="Literature Area")

    # Zoomed inset setup focused on the last data points
    axins = zoomed_inset_axes(ax, zoom=3, loc=2)  
    axins.errorbar(sample_sizes, uniform_areas, yerr=uniform_errors, color=colors[0], marker='o', linestyle='-', capsize=5)
    axins.errorbar(sample_sizes, hypercube_areas, yerr=hypercube_errors, color=colors[1], marker='s', linestyle='-', capsize=5)
    axins.errorbar(sample_sizes, ortho_areas, yerr=ortho_errors, color=colors[2], marker='^', linestyle='-', capsize=5)
    axins.errorbar(sample_sizes, masking_areas, yerr=masking_errors, color=colors[4], marker='x', linestyle='-', capsize=5)
    axins.axhline(y=lit_area, color=colors[3], linestyle='--')

    # Set the limits for the zoomed inset to focus on the maximum sample sizes
    last_points = masking_areas[-4:] + ortho_areas[-1:]
    x1, x2 = sample_sizes[-4]*0.9, sample_sizes[-1] * 1.1  # focus on the last few sample points, adding a bit of space
    y1, y2 = min(last_points) - 0.08, max(last_points) + 0.08
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # Hide ticks for the zoomed inset
    axins.set_xticks([])
    axins.set_yticks([])

    # Draw connecting lines between inset and main plot
    mark_inset(ax, axins, loc1=1, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel('Number of Samples', fontsize=20)
    ax.set_ylabel('Estimated Area', fontsize=20)
    ax.set_title(f'Estimated Area vs Number of Samples for {max_iter} iterations', fontsize=22)
    ax.legend(fontsize=18)
    ax.set_xscale('log')
    ax.grid(True)
    #plt.tight_layout()
    plt.savefig(f'Figures/Estimated_Area_vs_Number_of_Samples_{max_iter}_iterations.pdf', format='pdf')
    
    plt.show()
    return uniform_areas, uniform_errors, hypercube_areas, ortho_areas, ortho_errors

plot_comparison(sample_sizes, 200)

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