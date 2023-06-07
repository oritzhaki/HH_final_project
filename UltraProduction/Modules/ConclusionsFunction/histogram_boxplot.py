
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import os

def plot_histograms_and_boxplots(param_arr_good, param_arr_bad, result_path):
    # Create subplots for each parameter
    fig, axs = plt.subplots(4, 8, figsize=(24, 12))  # Added 2 more rows for boxplots
    
    # Define labels for the rows
    row_labels = ['Good Runs', 'Bad Runs']
    boxplot_labels = ['Good Runs Boxplot', 'Bad Runs Boxplot']
    
    # Number of bins
    n_bins = 30
    
    # Plot histograms and fitted Gaussian curves for each parameter
    for j, param_arr in enumerate([param_arr_good, param_arr_bad]):
        for i, arr in enumerate(param_arr):
            mu, std = norm.fit(arr)

            # Clip the data to be within 3 standard deviations from the mean
            clipped_arr = np.clip(arr, a_min=mu-3*std, a_max=mu+3*std)

            # Plot histogram
            n, bins, patches = axs[j,i].hist(clipped_arr, bins=n_bins, density=False, alpha=0.7, color='skyblue', edgecolor='black')

            # Calculate bin width
            bin_width = bins[1] - bins[0]

            # Plot fitted Gaussian curve, adjust for count scale and bin width
            xmin, xmax = axs[j,i].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std) * len(clipped_arr) * bin_width
            axs[j,i].plot(x, p, 'r', linewidth=2)

            # Add vertical line at the mean
            axs[j,i].axvline(mu, color='#FF00B3', linestyle='dashed', linewidth=2)

            # Set plot title and labels
            title_text = "Mean: {:.4f}".format(mu)
            title = axs[j,i].set_title(title_text, fontsize=10)
            title.set_color('#FF00B3')
            axs[j,i].set_xlabel('Value')
            axs[j,i].set_ylabel('Counts')

            # Label the rows
            if i == 0:
                axs[j,i].annotate(row_labels[j], xy=(0, 0.5), xytext=(-axs[j,i].yaxis.labelpad - 5, 0),
                                  xycoords=axs[j,i].yaxis.label, textcoords='offset points',
                                  size='large', ha='right', va='center')

            # Plot boxplots
            median = np.median(clipped_arr)
            median_str = "Median: {:.4f}".format(median)
            box = axs[j+2,i].boxplot(clipped_arr, vert=False, patch_artist=True)  # Plot horizontal boxplots
            axs[j+2,i].set_xlabel('Value')
            axs[j+2,i].set_title(median_str, fontsize=10)

            # Set colors
            box['boxes'][0].set_facecolor('skyblue')
            box['medians'][0].set_color('red')

            # Label the boxplot rows
            if i == 0:
                axs[j+2,i].annotate(boxplot_labels[j], xy=(0, 0.5), xytext=(-axs[j+2,i].yaxis.labelpad - 5, 0),
                                    xycoords=axs[j+2,i].yaxis.label, textcoords='offset points',
                                    size='large', ha='right', va='center')

    # Adjust spacing between subplots
    fig.tight_layout()

    images_path = os.path.join(result_path, 'Images')
    os.makedirs(images_path, exist_ok=True)

    # Save the plot
    plt.savefig(f'{images_path}/histogram_boxplot.png')
    print(f"Saved histogram_boxplot image in {images_path}")
