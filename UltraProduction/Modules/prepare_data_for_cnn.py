import os
import pandas as pd
import numpy as np

def prepare_data_for_cnn(base_dir):
    # Prepare an empty list to store the samples
    samples = []

    # Iterate over each KV directory
    for kv_dir in os.listdir(base_dir):
        kv_dir_path = os.path.join(base_dir, kv_dir)

        # Check if it's indeed a directory and starts with 'Kv1.1'
        if os.path.isdir(kv_dir_path) and kv_dir == 'Kv1.1':

            # Iterate over each Temp directory in the KV directory
            for temp_dir in os.listdir(kv_dir_path):
                temp_dir_path = os.path.join(kv_dir_path, temp_dir)

                # Check if it's indeed a directory
                if os.path.isdir(temp_dir_path):

                    # Iterate over each cell directory in the Temp directory
                    for cell_dir in os.listdir(temp_dir_path):
                        cell_dir_path = os.path.join(temp_dir_path, cell_dir)

                        # Check if it's indeed a directory
                        if os.path.isdir(cell_dir_path):

                            # Construct the path to the sample CSV file
                            csv_file_path = os.path.join(cell_dir_path, cell_dir + "_sample.csv")

                            # Check if the file exists
                            if os.path.isfile(csv_file_path):
                                
                                # Read the CSV file into a DataFrame
                                df = pd.read_csv(csv_file_path)
                                
                                # Extract the third column as a numpy array
                                array = df[df.columns[2]].values

                                # Determine the label of the cell
                                relative_path = csv_file_path.replace(base_dir + "/", "")
                                if relative_path.startswith('Kv1.1/Temp_15C'):
                                    label = 0
                                elif relative_path.startswith('Kv1.1/Temp_25C'):
                                    label = 1
                                elif relative_path.startswith('Kv1.1/Temp_35C'):
                                    label = 1
                                else:
                                    continue

                                # Add the sample to the list
                                samples.append([relative_path, array, label])

    return samples



def get_cell_numbers(cnn_data):
    return [item[0].split('/')[2] for item in cnn_data if item[0].split('/')[2] != 'dup']


def run():
    # The base directory containing the KV and Temp directories
    base_dir = "ConductivityData"
    
    return prepare_data_for_cnn(base_dir)

