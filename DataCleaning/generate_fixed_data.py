# This code:
# 1. Adding coloumns names
# 2. Deleting first 1000 and last 5000

import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
# Define the directory path where the data is stored
data_dir = "Data"

# Define the directory path where the modified data will be saved
new_data_dir = "NewData"

# Loop over each temperature directory
for temp_dir in os.listdir(data_dir):
    # Create the path to the temperature directory in the new data directory
    new_temp_dir_path = os.path.join(new_data_dir, temp_dir)
    os.makedirs(new_temp_dir_path, exist_ok=True)

    # Loop over each element directory in the temperature directory
    for elem_dir in os.listdir(os.path.join(data_dir, temp_dir)):
        # Create the path to the element directory in the new data directory
        new_elem_dir_path = os.path.join(new_temp_dir_path, elem_dir)
        os.makedirs(new_elem_dir_path, exist_ok=True)

        # Loop over each CSV file in the element directory
        for csv_file in os.listdir(os.path.join(data_dir, temp_dir, elem_dir)):
            # Create the path to the CSV file
            csv_file_path = os.path.join(data_dir, temp_dir, elem_dir, csv_file)
            print(csv_file)
            # Create the path to the new CSV file
            new_csv_file_path = os.path.join(new_elem_dir_path, csv_file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)
            df = df.applymap(lambda x: 0 if x < 0 else x)
            # Remove the first and last 1000 rows
            df = df.iloc[980:-5009]
            
            df = df[~((df.index % 10 != 0))]
            print(df.shape)

            # Add the column names to the data
            col_names = ["-90"]
            for i in range(1, len(df.columns)):
                col_names.append(str(int(col_names[i-1]) + 10))
            df.columns = col_names
            df = df.drop(df.columns[0], axis=1)
            # Write the modified data to a new CSV file
            df.to_csv(new_csv_file_path, index=False)
