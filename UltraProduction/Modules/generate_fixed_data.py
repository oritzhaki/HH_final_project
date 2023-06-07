# This code:
# 1. Adds column names
# 2. Deletes first 1000 and last 5000 rows

import os
import pandas as pd

import numpy as np

def generate_fixed_data(data_dir, target_dir):

    # Loop over each KV directory
    for kv_dir in os.listdir(data_dir):
        # Create the path to the KV directory in the new data directory
        new_kv_dir_path = os.path.join(target_dir, kv_dir)
        os.makedirs(new_kv_dir_path, exist_ok=True)

        # Loop over each temperature directory in the KV directory
        for temp_dir in os.listdir(os.path.join(data_dir, kv_dir)):
            # Create the path to the temperature directory in the new data directory
            new_temp_dir_path = os.path.join(new_kv_dir_path, temp_dir)
            os.makedirs(new_temp_dir_path, exist_ok=True)

            # Loop over each CSV file in the temperature directory
            for csv_file in os.listdir(os.path.join(data_dir, kv_dir, temp_dir)):
                # Create the path to the CSV file
                csv_file_path = os.path.join(data_dir, kv_dir, temp_dir, csv_file)
                
                # Create the path to the new CSV file
                new_csv_file_path = os.path.join(new_temp_dir_path, csv_file)

                # Load the CSV file into a DataFrame
                df = pd.read_csv(csv_file_path)
                df[df < 0] = 0  # Vectorized operation to replace negative values
                
                # Remove the first 980 rows and the last 5009 rows
                df = df.iloc[980:-5009]

                df = df.iloc[np.arange(0, df.shape[0], 10), :]  # More efficient row slicing

                # Add the column names to the data
                col_names = [str(i) for i in range(-90, 10 * len(df.columns) - 90, 10)]
                df.columns = col_names
                df = df.drop(df.columns[0], axis=1)
                
                if df.shape[0] == 100:
                    df.to_csv(new_csv_file_path, index=False)
                    print(f"Saved {new_csv_file_path} with shape {df.shape}")
                else:
                    print(f"Skipped {new_csv_file_path} with shape {df.shape}")


def run():
    # Define the directory path where the data is stored
    data_dir = "DataOrder"

    # Define the directory path where the modified data will be saved
    target_dir = "ConductivityData"
    
    generate_fixed_data(data_dir, target_dir)