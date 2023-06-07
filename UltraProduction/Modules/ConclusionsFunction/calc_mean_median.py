
import numpy as np

def calc_mean_median(df_good, df_bad, result_path):
    # initialize a list of 8 zeros to store the sum for each parameter
    param_sum = [0] * 8
    param_values = [[] for _ in range(8)]  # List to store values for calculating median

    # iterate over each row in the five columns (BestSol1, BestSol2, BestSol3, BestSol4, BestSol5)
    for col in ['BestSol1', 'BestSol2', 'BestSol3', 'BestSol4', 'BestSol5']:
        for x in df_good[col]:
            # convert the scientific notation string to a list of floats
            params = list(map(float, x.strip('[]').split()))
            # add the values of each parameter to the running total and store for calculating median
            for i in range(len(params)):
                param_sum[i] += params[i]
                param_values[i].append(params[i])

    # divide each sum by the number of rows and the number of columns to get the mean
    num_rows = len(df_good)
    num_cols = 5
    param_mean_good = [x / (num_rows * num_cols) for x in param_sum]

    # calculate the median for each parameter
    param_median_good = [np.median(values) for values in param_values]


    # initialize a list of 8 zeros to store the sum for each parameter
    param_sum = [0] * 8
    param_values = [[] for _ in range(8)]  # List to store values for calculating median

    # iterate over each row in the five columns (BestSol1, BestSol2, BestSol3, BestSol4, BestSol5)
    for col in ['BestSol1', 'BestSol2', 'BestSol3', 'BestSol4', 'BestSol5']:
        for x in df_bad[col]:
            # convert the scientific notation string to a list of floats
            params = list(map(float, x.strip('[]').split()))
            # add the values of each parameter to the running total and store for calculating median
            for i in range(len(params)):
                param_sum[i] += params[i]
                param_values[i].append(params[i])

    # divide each sum by the number of rows and the number of columns to get the mean
    num_rows = len(df_bad)
    num_cols = 5
    param_mean_bad = [x / (num_rows * num_cols) for x in param_sum]

    # calculate the median for each parameter
    param_median_bad = [np.median(values) for values in param_values]
    
    

    with open(f"{result_path}/mean_median.txt", "w") as f:
        f.write("Mean Good Results:\n")
        f.write(str(param_mean_good) + "\n")
        f.write("Median Good Results:\n")
        f.write(str(param_median_good) + "\n\n")

        f.write("Mean Bad Results:\n")
        f.write(str(param_mean_bad) + "\n")
        f.write("Median Bad Results:\n")
        f.write(str(param_median_bad) + "\n")
    
    return param_mean_good