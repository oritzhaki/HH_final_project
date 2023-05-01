# This code:
# 1. Remove Spikes
# 2. Using lowpass to clean data


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def remove_spikes(df):
    """
    Detects spikes in the input data array and replaces them with the average value of the two neighboring points.
    :param df: pandas dataframe containing the data to be cleaned.
    :return: cleaned pandas dataframe with the same shape as the input dataframe.
    """
    # Make a copy of the input dataframe
    df_cleaned = df.copy()

    # Iterate over each voltage column in the data array
    for i in range(1, df_cleaned.shape[1]):
        # Compute the difference between adjacent current values
        diff = np.diff(df_cleaned.iloc[:, i])

        # Identify the indices of spikes (i.e., where the difference is greater than 8 times the median difference)
        spike_indices = np.where(np.abs(diff) > 8 * np.median(np.abs(diff)))[0] + 1
        
        # Iterate over each spike index and replace the corresponding current value with the average of the two neighboring values
        for idx in spike_indices:
            # Check if the index is within the bounds of the dataframe
            if idx > 0 and idx < df_cleaned.shape[0] - 1:
                df_cleaned.iloc[idx, i] = (df_cleaned.iloc[idx - 1, i] + df_cleaned.iloc[idx + 1, i]) / 2

    # Return the cleaned dataframe
    return df_cleaned


def apply_low_pass_filter(df, cutoff_freq, sampling_freq, order=10):
    """
    Applies a low pass filter to each column in a pandas DataFrame.
    
    :param df: pandas DataFrame to filter
    :param cutoff_freq: cutoff frequency of the filter in Hz
    :param sampling_freq: sampling frequency of the data in Hz
    :param order: order of the filter (default=5)
    :return: filtered pandas DataFrame with the same shape as the input DataFrame
    """
    # Calculate the Nyquist frequency
    nyquist_freq = 0.5 * sampling_freq
    
    # Calculate the normalized cutoff frequency
    norm_cutoff_freq = cutoff_freq / nyquist_freq
    
    # Create a Butterworth filter with the specified order and cutoff frequency
    b, a = butter(order, norm_cutoff_freq, btype='lowpass')
    
    # Apply the filter to each column in the DataFrame using filtfilt to avoid phase shift
    filtered_data = filtfilt(b, a, df, axis=0)
    
    # Convert the filtered data to a pandas DataFrame with the same column names and index as the input DataFrame
    filtered_df = pd.DataFrame(filtered_data, columns=df.columns, index=df.index)
    
    return filtered_df


def show_plot(df, df_cleaned):
        # Extract the voltage values as a numpy array
        voltage_arr = df.columns[1:].astype(float)

        # Extract the current values as a numpy array
        current_arr = df.iloc[:, 0].values.astype(float)

        # Extract the time values as a numpy array (i.e., the row numbers)
        time_arr = np.arange(current_arr.shape[0])

        # Create an empty list to store the plot handles
        plot_handles = []

        # Plot the original data
        plt.subplot(1, 2, 1)
        for i in range(1, df.shape[1]):
            plot_handle, = plt.plot(time_arr, df.iloc[:, i].values.astype(float))
            plot_handles.append(plot_handle)

        # Add labels and title to the plot
        plt.xlabel("Time")
        plt.ylabel("Current")
        plt.title(f"Temperature {temp_dir} data (before cleaning)")

        # Add a legend to the right side of the plot
        plt.legend(plot_handles, voltage_arr, loc="center left", bbox_to_anchor=(1.05, 0.5), title="Voltage")

        # Adjust the figure size and add more space on the right side for the legend
        fig = plt.gcf()
        fig.set_size_inches(12, 5)
        plt.subplots_adjust(wspace=0.5)

        # Create an empty list to store the plot handles
        plot_handles = []

        # Plot the cleaned data
        plt.subplot(1, 2, 2)
        for i in range(1, df_cleaned.shape[1]):
            plot_handle, = plt.plot(time_arr, df_cleaned.iloc[:, i].values.astype(float))
            plot_handles.append(plot_handle)

        # Add labels and title to the plot
        plt.xlabel("Time")
        plt.ylabel("Current")
        plt.title(f"Temperature {temp_dir} data (after cleaning)")

        # Add a legend to the right side of the plot
        plt.legend(plot_handles, voltage_arr, loc="center left", bbox_to_anchor=(1.05, 0.5), title="Voltage")

        # Adjust the figure size and add more space on the right side for the legend
        fig = plt.gcf()
        fig.set_size_inches(12, 5)
        plt.subplots_adjust()

        # Show the plot
        plt.show()

# Define the directory path where the modified data is stored
data_dir = "NewData"
clean_data_dir = "CleanData"
LEVEL = 5

# Loop over each temperature directory
for temp_dir in os.listdir(data_dir):
    # Create the path to the temperature directory
    temp_dir_path = os.path.join(data_dir, temp_dir)

    # Loop over each CSV file in the temperature directory
    for csv_file in os.listdir(temp_dir_path):
        # Create the path to the CSV file
        csv_file_path = os.path.join(temp_dir_path, csv_file, f'{csv_file}_Activation_rep2.csv')
        print(csv_file_path)

        # Read the CSV file and store the data in a pandas dataframe
        df = pd.read_csv(csv_file_path)

        # Clean the data using the remove_spikes function
        df_cleaned = remove_spikes(df)
        
        for i in range(LEVEL):
            df_cleaned = remove_spikes(df_cleaned)
        
        df_cleaned = apply_low_pass_filter(df_cleaned, 20, 300)

        # Create the path to the new CSV file
        new_csv_file_path = os.path.join(clean_data_dir, temp_dir, csv_file, f'{csv_file}_Activation_rep2.csv')

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(new_csv_file_path), exist_ok=True)

        # Save the cleaned data to a new CSV file
        df_cleaned.to_csv(new_csv_file_path, index=False)
        
        
        ################################################ SHOW PLOT ###########################################
        
        # show_plot(df, df_cleaned)