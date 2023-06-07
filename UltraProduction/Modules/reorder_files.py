# reorder_files.py

import os
import shutil

def copy_files(source_dir, target_dir):
    # Traverse the directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if it's a csv file
            if file.endswith('.csv'):
                # Create the new filename
                new_filename = file.split('_')[0] + '.csv'
                
                # Get the old file path
                old_file_path = os.path.join(root, file)
                
                # Define the new file path
                new_file_path = os.path.join(target_dir, *root.split('/')[-3:-1])
                
                # Create new directories if they do not exist
                os.makedirs(new_file_path, exist_ok=True)
                
                # Define the final new file path
                final_new_file_path = os.path.join(new_file_path, new_filename)
                
                # Copy the file to the new path
                shutil.copy2(old_file_path, final_new_file_path)

                print(f"Copied {old_file_path} to {final_new_file_path}")


def run():
    # Define your source and target directories
    script_dir = os.path.dirname(os.path.realpath(__file__))
    source_dir = os.path.join(script_dir, '../Data')
    target_dir = os.path.join(script_dir, '../DataOrder')

    # Call the function
    copy_files(source_dir, target_dir)
