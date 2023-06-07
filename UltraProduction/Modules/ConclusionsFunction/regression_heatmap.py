

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


def numpy_string_to_list(numpy_string):
    numpy_string = numpy_string.strip('[]')  # remove brackets
    string_list = numpy_string.split()  # split by spaces
    return list(map(float, string_list))  # convert to floats and return as a list

def plot_regression(result_path):
    df = pd.read_csv(f'{result_path}/GoodRuns.csv')


    # First, we need to convert string representations of lists into actual list objects
    df['BestSol1'] = df['BestSol1'].apply(numpy_string_to_list)

    # Now, we can split the 'BestSol1' column into separate columns for each parameter
    params = df['BestSol1'].apply(pd.Series)
    params.columns = [f'c[{i}]' for i in range(8)]

    # Now we compute the correlation matrix.
    corr = params.corr()

    pairplot_figure = sns.pairplot(params, diag_kind='False')

    # Save the figure to a PNG file
    images_path = os.path.join(result_path, 'Images')
    os.makedirs(images_path, exist_ok=True)
    pairplot_figure.savefig(f'{images_path}/regression_plot.png')
    print(f"Saved regression_plot image in {images_path}")
    return corr



def plot_heatmap(corr, result_path):
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=False, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    images_path = os.path.join(result_path, 'Images')
    os.makedirs(images_path, exist_ok=True)

    # Save the plot
    plt.savefig(f'{images_path}/plot_heatmap.png')
    print(f"Saved plot_heatmap image in {images_path}")