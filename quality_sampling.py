import mandelbrot
import matplotlib.pyplot as plt
import numpy as np
import Sample_methods as sm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

sample_sizes = [10**2, 20**2, 30**2, 40**2, 50**2, 60**2, 70**2, 80**2, 90**2, 100**2, 200**2, 300**2, 400**2, 500**2, 600**2, 700**2, 800**2, 900**2, 1000**2] #, 5000**2]	#10e8
lit_area = 1.5052 # Literature value
font_size = 8
marker_size = 3
cap = 2
lw = 0.8
alpha = 0.6
colors = ['tab:blue', 'tab:green', 'tab:red', "tab:purple", "tab:orange"]
# Use LaTex font for labels
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
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
    fig, ax = plt.subplots(figsize=(5.91, 3.6))
    ax.errorbar(sample_sizes, uniform_areas, yerr=uniform_errors, label='Uniform sampling', color=colors[0], marker='s', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    ax.errorbar(sample_sizes, hypercube_areas, yerr=hypercube_errors, label='Hypercube sampling', color=colors[1], marker='o', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    ax.errorbar(sample_sizes, ortho_areas, yerr=ortho_errors, label='Orthogonal sampling', color=colors[2], marker='^', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    ax.errorbar(sample_sizes, masking_areas, yerr=masking_errors, label='Custom sampling', color=colors[4], marker='x', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    ax.axhline(y=lit_area, color=colors[3], linestyle='--', label="$A_{M} = 1.5052$")

    # Zoomed figure for the last few sample sizes
    axins = zoomed_inset_axes(ax, zoom=5, loc=4, bbox_to_anchor=(1.015, 0.798), bbox_transform=ax.transAxes)  
    axins.errorbar(sample_sizes, uniform_areas, yerr=uniform_errors, color=colors[0], marker='s', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    axins.errorbar(sample_sizes, hypercube_areas, yerr=hypercube_errors, color=colors[1], marker='o', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    axins.errorbar(sample_sizes, ortho_areas, yerr=ortho_errors, color=colors[2], marker='^', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    axins.errorbar(sample_sizes, masking_areas, yerr=masking_errors, color=colors[4], marker='x', linestyle='-', capsize = cap, markersize=marker_size, linewidth=lw, alpha = alpha)
    axins.axhline(y=lit_area, color=colors[3], linestyle='--', alpha = 0.8, linewidth = 0.8)

    # Define the limits for the zoomed inset (last few sample sizes)
    last_points = masking_areas[-7:] + ortho_areas[-1:]
    x1, x2 = sample_sizes[-7]*0.9, sample_sizes[-1] * 1.02 
    y1, y2 = min(last_points) - 0.025, max(last_points) + 0.025
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    # Hide the ticks for the zoomed inset
    axins.set_xticks([])
    axins.set_yticks([])

    # Draw connecting lines between inset and main plot
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5", lw=0.5) 
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", lw=0.5)
    ax.tick_params(axis='x', labelsize=font_size)  
    ax.tick_params(axis='y', labelsize=font_size)
    ax.set_ylim(0.8, 2.36)
    ax.set_xlabel('Number of Samples', fontsize=font_size)
    ax.set_ylabel('Estimated Area', fontsize=font_size)
    #    ax.set_title(f'Performance sampling methods \n({max_iter} iterations)', fontsize=32)
    ax.legend(fontsize=font_size, loc= 'lower right')
    ax.set_xscale('log')
    ax.grid(True)
    plt.savefig(f'Figures/Estimated_Area_vs_Number_of_Samples_{max_iter}_iterations.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    return uniform_areas, uniform_errors, hypercube_areas, hypercube_errors, ortho_areas, ortho_errors, masking_areas, masking_errors

plot_comparison(sample_sizes, 150)
