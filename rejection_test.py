import mandelbrot
import numpy as np
import time
import pandas as pd
import Rejection

def test_rejection_stochasticity(trials=10):
    np.random.seed(None)
    img_sizes = [10, 50, 100]
    i_spaces = [5, 8, 10]
    Z_boundaries = [0.5, 0.8, 0.9, 1.0]
    sample_size = 1000  # Fixed for comparison

    results = []
    for img_size in img_sizes:
        for i_space in i_spaces:
            for Z_boundary in Z_boundaries:
                areas = []
                times = []
                for _ in range(trials):
                    start_time = time.time()
                    area_total, _, samples = Rejection.rejection(img_size, i_space, Z_boundary, sample_size)
                    evaluations = []
                    for x, y in samples:
                        eval = mandelbrot.eval_point_mandelbrot(x, y, sample_size) == 1
                        evaluations.append(eval)

                    area = area_total * sum(evaluations) / len(evaluations)

                    elapsed_time = time.time() - start_time
                    areas.append(area)
                    times.append(elapsed_time)
                
                # Compute mean and standard deviation
                mean_area = np.mean(areas)
                std_area = np.std(areas, ddof=1)  # Sample standard deviation
                mean_time = np.mean(times)

                results.append({
                    'img_size': img_size,
                    'i_space': i_space,
                    'Z_boundary': Z_boundary,
                    'mean_area': mean_area,
                    'std_area': std_area,
                    'mean_time': mean_time
                })

    df = pd.DataFrame(results)
    print(df.sort_values(by='std_area'))

test_rejection_stochasticity()