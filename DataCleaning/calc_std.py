import pandas as pd
import numpy as np

# Initialize a list to store the DataFrames
dataframes = []

# Read the first CSV file to get the column headers
first_file_path = "dataset/sample_1.csv"
first_df = pd.read_csv(first_file_path)
column_headers = first_df.columns

# Iterate over the range of file numbers
for i in range(1, 11):
    # Construct the file path
    file_path = f"dataset/sample_{i}.csv"

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate the DataFrames along a new axis (resulting in a 3D array)
arr = np.stack(dataframes, axis=2)

# Calculate the standard deviation along the third axis
std_arr = np.std(arr, axis=2)

# Create a new DataFrame from the standard deviation array
result_df = pd.DataFrame(std_arr, columns=column_headers)

# Display the result DataFrame
print(result_df)

# Save the result DataFrame to a CSV file
result_df.to_csv("std_graph_df.csv", index=False)

import pandas as pd

# Read the "std_graph_df.csv" file
df = pd.read_csv('std_graph_df.csv')

# Get the number of rows and columns in the original DataFrame
num_rows, num_cols = df.shape

# Initialize an empty list to store the new rows
new_rows = []

# Iterate over the columns of the original DataFrame
for col_idx in range(num_cols):
    # Get the column name and corresponding standard deviation values
    col_name = df.columns[col_idx]
    std_values = df[col_name].values

    # Iterate over the rows and add the row number, column name, and standard deviation value to the new rows list
    for row_idx, std_value in enumerate(std_values):
        new_row = [row_idx, col_name, std_value]
        new_rows.append(new_row)

# Create a new DataFrame from the new rows list
result_df = pd.DataFrame(new_rows)

# Display the result DataFrame
print(result_df)

# Save the result DataFrame to a CSV file
result_df.to_csv('std.csv', index=False)
