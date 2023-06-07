import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_kde(param_arr_good, param_arr_bad, result_path):
    # Create subplots for each parameter
    fig, axs = plt.subplots(2, 8, figsize=(24, 6))
    
    # Define labels for the rows
    row_labels = ['Good Runs', 'Bad Runs']
    
    # Plot KDEs for each parameter
    for j, param_arr in enumerate([param_arr_good, param_arr_bad]):
        for i, arr in enumerate(param_arr):
            # Calculate mean
            mu = np.mean(arr)

            # Plot KDE
            sns.kdeplot(arr, ax=axs[j,i], fill=True, warn_singular=False)

            # Add vertical line at the mean
            axs[j,i].axvline(mu, color='orange', linestyle='dashed', linewidth=3)

            # Set plot title and labels
            title_text = "Mean: {:.4f}".format(mu)
            title = axs[j,i].set_title(title_text, fontsize=10)
            title.set_color('blue')
            axs[j,i].set_xlabel('Value')
            axs[j,i].set_yticks([]) # This line removes the y-axis labels

            # Label the rows
            if i == 0:
                axs[j,i].annotate(row_labels[j], xy=(0, 0.5), xytext=(-axs[j,i].yaxis.labelpad - 5, 0),
                                  xycoords=axs[j,i].yaxis.label, textcoords='offset points',
                                  size='large', ha='right', va='center')

    # Adjust spacing between subplots
    fig.tight_layout()

    images_path = os.path.join(result_path, 'Images')
    os.makedirs(images_path, exist_ok=True)

    # Save the plot
    plt.savefig(f'{images_path}/kde.png')
    print(f"Saved kde image in {images_path}")
