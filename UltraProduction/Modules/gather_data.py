import os
import pandas as pd
import numpy as np
def get_data_paths(base_dir):
    # Prepare an empty list to store the paths
    paths = []

    # Iterate over each KV directory
    for kv_dir in os.listdir(base_dir):
        kv_dir_path = os.path.join(base_dir, kv_dir)

        # Check if it's indeed a directory
        if os.path.isdir(kv_dir_path):

            # Iterate over each Temp directory in the KV directory
            for temp_dir in os.listdir(kv_dir_path):
                temp_dir_path = os.path.join(kv_dir_path, temp_dir)

                # Check if it's indeed a directory
                if os.path.isdir(temp_dir_path):

                    # Add the directory path to the list
                    paths.append(temp_dir_path)

    return paths

def gather_paths():
    # The base directory containing the KV and Temp directories
    base_dir = "ConductivityData"
    
    data_paths = get_data_paths(base_dir)
    return data_paths
    
    
    
    
    
    
    
    
    
    
    
    
def read_csvs_from_dir(root_dir):
    data = [] # list to save tuples (cell_num, numpy array)

    # walk through directory hierarchy
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if '_sample.csv' in filename: # check if file is a sample CSV
                cell_num = os.path.basename(dirpath) # extract cell number from folder name
                file_path = os.path.join(dirpath, filename) # get full file path
                df = pd.read_csv(file_path, usecols=[2]) # read only third column into dataframe
                np_arr = df.values.flatten() # convert dataframe to numpy array
                data.append((cell_num, np_arr)) # append tuple to list

    return data
    
    
    
    
    
    
def calculate_change_rate_score(cell_data):
    # Calculate the slope between each pair of consecutive points
    slopes = np.diff(cell_data, axis=0) / np.diff(np.arange(len(cell_data)))

    # Calculate the absolute difference between the slopes of consecutive points
    slope_diffs = np.abs(np.diff(slopes, axis=0))

    # Return the average of these differences as the "change rate score"
    return np.mean(slope_diffs)
    

def evaluate_cells(data):
    dataclone = []
    # Calculate change rate score for each cell and add it to the data
    for i in range(len(data)):
        cell_num, cell_data = data[i]
        change_rate_score = calculate_change_rate_score(cell_data)
        dataclone.append([cell_num, cell_data, change_rate_score])

    # Sort cells by change rate score (lower = smoother)
    dataclone.sort(key=lambda x: x[2])
    return dataclone[0:2]






def get_best_cells(root_dir):
    data = read_csvs_from_dir(root_dir)
    best_cells = evaluate_cells(data)
    return best_cells
    
