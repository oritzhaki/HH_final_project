import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

os.chdir('C:\\Users\\galle\\OneDrive\\Desktop\\Directories\\Study\\ShanaC\\Project\\Git_HH\\DataCleaning')

CHOSEN_ROW = 70
VOLT_ARR = [-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60 ,70, 80, 90]




# Define the directory path where the modified data is stored
data_dir = "NewData"
conductivity_data_dir = "CleanConductivityData"

def get_prediction(df, df_num):
    if (df_num == '8021' or df_num == '9394' or df_num == '9398' or df_num == '9403' or df_num == '9406' or df_num == '9409' or df_num == '9412'):
       df.to_csv(f'GoodConductivityData/{df_num}.csv', index=False) 


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

# Loop over each temperature directory
for temp_dir in os.listdir(data_dir):
    # Create the path to the temperature directory
    temp_dir_path = os.path.join(data_dir, temp_dir)

    # Loop over each CSV file in the temperature directory
    for csv_file in os.listdir(temp_dir_path):
        # Create the path to the CSV file
        csv_file_path = os.path.join(temp_dir_path, csv_file, f'{csv_file}_Activation_rep2.csv')
        print(temp_dir)

        # Read the CSV file and store the data in a pandas dataframe
        df = pd.read_csv(csv_file_path)
        # df = df.iloc[10:-1]
        cut = generate_cut(df)
        slope, b = calc_linear_regression(cut)
        dots = create_subtract_dots(slope, b)
        df_after_substract = subtract_dots_from_dF_current(df, dots)
        conductivity_df = create_cunductivity_dF(df_after_substract, temp_dir)
        conductivity_df = conductivity_df.drop(conductivity_df.columns[0], axis=1)
        high_value = generate_high_value(conductivity_df)
        conductivity_df = create_cunductivity_dF_normalize_by_high_value(conductivity_df, high_value)
        #clean_conductivity_df = apply_low_pass_filter(conductivity_df, 10, 100)
        clean_conductivity_df = conductivity_df.applymap(lambda x: 0 if x < 0 else x)
        
        get_prediction(clean_conductivity_df, csv_file)
        
        # clean_conductivity_df.plot()
        # plt.title(f'{csv_file}')
        # plt.show()