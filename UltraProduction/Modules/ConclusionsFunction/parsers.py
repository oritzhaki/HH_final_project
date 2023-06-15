import numpy as np
import pandas as pd
import math
import os

def get_result_path(path):
    # Split the path to get the directory details
    path_parts = path.split('/')
    y_y, z_c, x, _ = path_parts
    return f"Results/{y_y}/{z_c}/{x}"


import os

def get_relevant_path(path):
    # Split the path to get the directory and file details
    parent_dir, filename = os.path.split(path)

    # We use os.path.split to get the last part of the path (cell number)
    _, cell = os.path.split(parent_dir)

    # We use os.path.dirname twice more to get the parent of the parent directory
    temp_dir = os.path.dirname(parent_dir)

    # We use os.path.split to get the last part of the path (Temp and Kv1.1)
    _, temp = os.path.split(temp_dir)
    _, kv = os.path.split(os.path.dirname(temp_dir))

    return f"{kv}/{temp}/{cell}/{filename}"


def modified_euclidean_distance(x, y):
    distance = 0
    for i in range(0,len(x)):
        if x[i] == 0.0:
            x[i] = 1e-10
        if y[i] == 0.0:
            y[i] = 1e-10
        r = ((x[i] - y[i]) ** 2) * (max(x[i]/y[i], y[i]/x[i]) ** 2)
        distance += r

    return math.sqrt(distance)


def scientific_to_float(scientific_notation):
    parts = scientific_notation.split('e')
    if len(parts) == 1:
        return float(parts[0])
    else:
        a, b = parts[0], int(parts[1])
        if '.' in a:
            c = len(a) - a.index('.') - 1
            a = a.replace('.', '')
            b -= c
        if b < 0:
            return float(a) / (10 ** abs(b))
        else:
            return float(a) * (10 ** b)
        
# Function to process the data
def process_data(file_name):
    # read CSV file
    df = pd.read_csv(file_name)

    # initialize empty NumPy arrays for each parameter
    param_arr = [np.empty(len(df)) for _ in range(8)]

    # iterate over each row in the five columns (BestSol1, BestSol2, BestSol3, BestSol4, BestSol5)
    for col in ['BestSol1', 'BestSol2', 'BestSol3', 'BestSol4', 'BestSol5']:
        for row_idx, x in enumerate(df[col]):
            # convert the scientific notation string to a list of floats
            params = list(map(float, x.strip('[]').split()))
            # assign each parameter to the corresponding NumPy array
            for i in range(8):
                param_arr[i][row_idx] = params[i]

    for i in range(len(param_arr)):
        param_arr[i] = np.copy(param_arr[i])
        param_arr[i] = np.delete(param_arr[i], np.where(param_arr[i] < 0))

    return param_arr