import os
import pandas as pd

def transform_df(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # Get the number of rows and columns in the original DataFrame
    num_rows, num_cols = df.shape

    # Initialize an empty list to store the new rows
    new_rows = []

    # Iterate over the columns of the original DataFrame
    for col_idx in range(num_cols):
        # Get the column name and corresponding values
        col_name = df.columns[col_idx]
        values = df[col_name].values

        # Iterate over the rows and add the row number, column name, and value to the new rows list
        for row_idx, value in enumerate(values):
            new_row = [row_idx, col_name, value]
            new_rows.append(new_row)

    # Create a new DataFrame from the new rows list
    reshaped_df = pd.DataFrame(new_rows, columns=['0', '1', '2'])

    return reshaped_df

def process_files_in_directory(data_dir):
    # Iterate over each directory in the ConductivityData directory
    for kv_dir in os.listdir(data_dir):
        kv_dir_path = os.path.join(data_dir, kv_dir)
        
        # Check if it's indeed a directory
        if os.path.isdir(kv_dir_path):
            
            for temp_dir in os.listdir(kv_dir_path):
                temp_dir_path = os.path.join(kv_dir_path, temp_dir)
                
                # Again, check if it's indeed a directory
                if os.path.isdir(temp_dir_path):
                    
                    # Iterate over each CSV file in the directory
                    for csv_file in os.listdir(temp_dir_path):
                        # Create the path to the CSV file
                        csv_file_path = os.path.join(temp_dir_path, csv_file)

                        # Check if it's indeed a file
                        if os.path.isfile(csv_file_path):
                            # Get the base name of the csv file (without extension)
                            base_name = os.path.splitext(csv_file)[0]
                            
                            # Create a new directory with the same name as the CSV file
                            new_dir_path = os.path.join(temp_dir_path, base_name)
                            os.makedirs(new_dir_path, exist_ok=True)
                            
                            # Copy the original CSV file to the new directory
                            os.system(f'cp {csv_file_path} {os.path.join(new_dir_path, base_name + "_graph.csv")}')
                            
                            # Apply transformation
                            transformed_df = transform_df(csv_file_path)
                            
                            # Save the transformed DataFrame to a new CSV file in the new directory
                            transformed_df.to_csv(os.path.join(new_dir_path, base_name + "_sample.csv"), index=False)
                            
                            # Remove the original CSV file
                            os.remove(csv_file_path)
                            
                            print(f"transformation has been done for {base_name}")


def run():
    # Define the directory path where the data is stored
    data_dir = "ConductivityData"
    process_files_in_directory(data_dir)

