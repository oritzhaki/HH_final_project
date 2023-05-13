import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fetch_plot(i):

    # Read the dataset CSV into a DataFrame
    dataset_df = pd.read_csv(f'Noise_Data/dataset_noise_{i}.csv', header=None)

    # Get the number of rows and columns in the dataset
    num_rows = dataset_df.shape[0]
    num_cols = dataset_df.shape[1]

    # Calculate the number of batches
    num_batches = num_rows // 100

    # Create an empty DataFrame for the new shape
    plot_df = pd.DataFrame(columns=range(num_batches))
    header = []
    # Iterate over the batches and populate the new DataFrame
    for i in range(num_batches):
        start_row = i * 100
        end_row = (i + 1) * 100

        # Get the column header name from the second column of the first row in the batch
        col_header = dataset_df.iloc[start_row + 1, 1]
        header.append(col_header)
        # Get the values for the batch from the third column
        batch_values = dataset_df.iloc[start_row:end_row, 2].values
        # print(batch_values)
        # Set the values as a new column in the new DataFrame
        plot_df[i] = batch_values

    # Set the column names based on the second column of the first row in each batch
    plot_df.columns = header
    plot_df = plot_df.drop(0)
    # Save the new DataFrame to a CSV file
    # new_df.to_csv('new_dataset.csv', index=False)

    plot_df.plot()
    plt.legend(np.arange(-70, 100, 10))
    plt.xlabel("Time (ms)")
    plt.ylabel("Conductivity (G)")
    plt.show()
    

for i in range(1,11):
    fetch_plot(i)