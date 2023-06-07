import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

CHOSEN_ROW = 70
VOLT_ARR = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60 ,70, 80, 90]

def current_to_conductivity(data_dir):

    # Define the directory path where the modified data is stored
    data_dir = "ConductivityData"

    def apply_low_pass_filter(df, cutoff_freq, sampling_freq, order=10):
        nyquist_freq = 0.5 * sampling_freq
        norm_cutoff_freq = cutoff_freq / nyquist_freq
        b, a = butter(order, norm_cutoff_freq, btype='lowpass')
        filtered_data = filtfilt(b, a, df, axis=0)
        filtered_df = pd.DataFrame(filtered_data, columns=df.columns, index=df.index)
        return filtered_df

    def get_temp(str):
        temperature_str = str.split("_")[-1][:-1]
        temperature = float(temperature_str)
        return temperature

    def generate_Ek(str):
        temperature = get_temp(str)
        Ek = -0.285 * (temperature + 273)
        return Ek

    def generate_cut(df):
        cut = df.loc[CHOSEN_ROW, :].values.tolist()
        return cut

    def calc_linear_regression(cut):
        y2 = cut[2]
        y1 = cut[0]
        x2 = -60
        x1 = -90
        slope = ((y2 - y1) / (x2 - x1))
        b = y2 - (slope * x2)
        return slope, b

    def create_subtract_dots(slope ,b):
        dots_Arr = []
        for v in VOLT_ARR:
            I = (slope * v) + b
            dots_Arr.append(I)
        return dots_Arr

    def subtract_dots_from_dF_current(df, dots):
        return df.subtract(dots, axis = 1)

    def create_cunductivity_dF(df_after_substract, temp):
        i = 0
        cunductivity_dF = df_after_substract.copy()
        for column in df_after_substract:
            col_to_add = df_after_substract[column]
            cunductivity_dF[column] = col_to_add.div(VOLT_ARR[i] - generate_Ek(temp))
            i+=1
        return cunductivity_dF

    def generate_high_value(conductivity_df):
        last_col = conductivity_df.iloc[ : , -1:].values
        return max(last_col)

    def create_cunductivity_dF_normalize_by_high_value(conductivity_df, high_value):
        return conductivity_df.div(high_value[0])

    # Loop over each KV directory
    for kv_dir in os.listdir(data_dir):
        # Create the path to the KV directory
        kv_dir_path = os.path.join(data_dir, kv_dir)

        # Loop over each temperature directory in the KV directory
        for temp_dir in os.listdir(kv_dir_path):
            # Create the path to the temperature directory
            temp_dir_path = os.path.join(kv_dir_path, temp_dir)

            # Loop over each CSV file in the temperature directory
            for csv_file in os.listdir(temp_dir_path):
                # Create the path to the CSV file
                csv_file_path = os.path.join(temp_dir_path, csv_file)

                # Read the CSV file and store the data in a pandas dataframe
                df = pd.read_csv(csv_file_path)

                # Preprocess the data
                cut = generate_cut(df)
                slope, b = calc_linear_regression(cut)
                dots = create_subtract_dots(slope, b)
                df_after_substract = subtract_dots_from_dF_current(df, dots)
                conductivity_df = create_cunductivity_dF(df_after_substract, temp_dir)
                conductivity_df = conductivity_df.drop(conductivity_df.columns[0], axis=1)
                high_value = generate_high_value(conductivity_df)
                conductivity_df = create_cunductivity_dF_normalize_by_high_value(conductivity_df, high_value)
                # clean_conductivity_df = conductivity_df.applymap(lambda x: 0 if x < 0 else x)

                conductivity_df = conductivity_df.drop(conductivity_df.columns[:5], axis=1)
                # Save the cleaned data to the original CSV file, overriding the original data
                conductivity_df.to_csv(csv_file_path, index=False)

                print(f"Processed and saved {csv_file_path}")



def run():
    # Define your source and target directories
    script_dir = os.path.dirname(os.path.realpath(__file__))
    target_dir = os.path.join(script_dir, '../ConductivityData')

    # Call the function
    current_to_conductivity(target_dir)
