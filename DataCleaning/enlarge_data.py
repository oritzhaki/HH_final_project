import os
import pandas as pd
import numpy as np


os.chdir('C:\\Users\\galle\\OneDrive\\Desktop\\Directories\\Study\\ShanaC\\Project\\Git_HH\\DataCleaning')


# Set the directory containing the CSV files
directory = "GoodConductivityData/"

# Create the directory to save the new datasets
new_directory = "dataset/"
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# Create an empty list to store the dataframes
dataframes = []

# Loop through all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Load the CSV file as a dataframe
        df = pd.read_csv(os.path.join(directory, filename))
        # Append the dataframe to the list
        dataframes.append(df)

# Loop through 100 iterations to generate new datasets
for i in range(1, 101):
    # Generate a new list of random fractions of length equal to the number of dataframes
    rand_fractions = np.random.dirichlet(np.ones(len(dataframes)), size=1)[0]

    # Loop through each dataframe and multiply each cell by the corresponding fraction
    arrays = []
    for j, df in enumerate(dataframes):
        array = np.multiply(df.values, rand_fractions[j])
        arrays.append(array)

    # Sum the arrays and convert to a dataframe
    new_array = np.sum(arrays, axis=0)
    new_df = pd.DataFrame(new_array, columns=dataframes[0].columns)

    # Save the new dataframe as a CSV file
    new_filename = os.path.join(new_directory, f"sample_{i}.csv")
    new_df.to_csv(new_filename, index=False)
