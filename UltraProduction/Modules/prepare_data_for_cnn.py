import os
import pandas as pd
import numpy as np

def prepare_data_for_cnn(base_dir, cells_with_label_1):

    # Prepare an empty list to store the samples
    samples = []

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
                                label = 1 if cell_dir in cells_with_label_1 else 0

                                # Add the sample to the list
                                samples.append([csv_file_path.replace(base_dir + "/", ""), array, label])

    # Filter the samples for the ones in the 'Kv1.1/Temp_15C' directory
    samples_Kv1_1_15C = list(filter(lambda sample: sample[0].startswith('Kv1.1/Temp_15C'), samples))

    # Count number of samples in each class
    class_counts = {0: 0, 1: 0}
    for sample in samples_Kv1_1_15C:
        class_counts[sample[2]] += 1

    # Determine which class is the minority
    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)

    # Calculate the amount of duplication required for each sample in the minority class
    duplication_factor = class_counts[max_class] // class_counts[min_class]
    # duplication_factor = 3

    # Create new unique instances for the minority class
    duplicated_samples = []
    for i in range(duplication_factor - 1):
        for s in samples_Kv1_1_15C:
            if s[2] == min_class:
                # Create a new path for the duplicated instance
                new_path = 'dup/dup/dup'

                # Create a copy of the numpy array
                array = np.copy(s[1])

                # Calculate the number of elements that will have noise added
                num_noisy_elements = int(0.10 * len(array))

                # Randomly select indices of the array to add noise to
                noisy_indices = np.random.choice(len(array), num_noisy_elements, replace=False)

                # Generate random noise
                noise = np.random.uniform(-0.001, 0.001, num_noisy_elements)

                # Add the noise to the selected elements of the array
                array[noisy_indices] += noise

                # Create a new instance with the new path and noisy array
                new_instance = [new_path, array, s[2]]

                # Add the new instance to the duplicated samples
                duplicated_samples.append(new_instance)

    # Add the duplicated instances to the main list
    samples_Kv1_1_15C += duplicated_samples



    # Add the duplicated instances to the main list
    samples_Kv1_1_15C += duplicated_samples

    return samples_Kv1_1_15C


def get_cell_numbers(cnn_data):
    return [item[0].split('/')[2] for item in cnn_data if item[0].split('/')[2] != 'dup']


def run():
    # The base directory containing the KV and Temp directories
    base_dir = "ConductivityData"

    # List of cells that have label 1
    cells_with_label_1 = ["8021", "9403"]
    
    return prepare_data_for_cnn(base_dir, cells_with_label_1)
