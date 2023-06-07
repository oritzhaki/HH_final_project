
import numpy as np
import pandas as pd
import os

def alpha_n(V, C):
    result_alpha = (C[0] * (V + C[1])) / (1 - np.exp(-C[2] * (V + C[3])))
    return result_alpha

def beta_n(V, C):
    result_beta = C[4] * (np.exp(-C[5] * ( V + C[6] )))
    return result_beta

def n_inf(alpha, beta):
    result_n_inf =  alpha / (alpha + beta)
    return result_n_inf

def tau_n(alpha, beta):
    result_tau = 1 / (alpha + beta)
    return result_tau

def n_pow_4(n, C):
    result_n =  n ** C[7]
    return result_n
    
def get_y(t, V, C):
    alpha = alpha_n(V,C)
    beta = beta_n(V,C)
    n_inf_ = n_inf(alpha, beta)
    tau_n_ = tau_n(alpha, beta)
    n = n_inf_ * (1 - np.exp(-t/tau_n_))
    y_hat = n_pow_4(n,C) 
    return y_hat


def generate_data_base_param(result_path, param_mean_good):
    C = param_mean_good
    dataset = []
    n = 0
    t_total = 100
    VOLTS = np.arange(-20, 100, 10)
    results = []

    for V in VOLTS:
        voltage_results = []
        for t in np.arange(0, t_total):
            y = get_y(t, V, C)
            voltage_results.append(y)
            dataset.append((t, V, y))
        results.append(voltage_results)


    graph_df = pd.DataFrame(results)
    graph_df = graph_df.T
    dataset_df = pd.DataFrame(dataset)
    graph_df.plot()

    # Ensure the directory exists
    generated_data_path = os.path.join(result_path, 'GeneratedDataBasedGAParams')
    os.makedirs(generated_data_path, exist_ok=True)

    # Save the dataframes
    graph_df.to_csv(f'{generated_data_path}/graph_df_based_ga_params.csv', index=False)
    dataset_df.to_csv(f'{generated_data_path}/dataset_based_ga_params.csv', index=False)