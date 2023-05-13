import pandas as pd
import numpy as np

STD_FACTOR = 5

def add_noise(i):
    # Read the "dataset.csv" and "std.csv" files into DataFrames
    dataset_df = pd.read_csv('dataset.csv', header=None)
    std_df = pd.read_csv('std.csv', header=None)

    # Extract the values from the "std" DataFrame
    std_values = std_df.iloc[:, 2].values

    # Iterate over the rows of the "dataset" DataFrame
    for idx, row in dataset_df.iterrows():
        # Get the corresponding standard deviation value
        std_value = std_values[idx]

        # Calculate a random number between the negative and positive values
        min_value = -std_value
        max_value = std_value
        num = np.random.uniform(min_value, max_value)

        # Add the calculated number to the value in the corresponding cell of the "dataset" DataFrame
        dataset_df.at[idx, 2] += (num / STD_FACTOR)

    # Save the updated DataFrame to a new CSV file
    dataset_df.columns = [0, 1, 2]
    dataset_df = dataset_df.drop(0)
    dataset_df.to_csv(f'Noise_Data/dataset_noise_{i}.csv', header=True, index=False)

for i in range(1,11):
    add_noise(i)