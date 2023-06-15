
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import os
from Modules.ConclusionsFunction.parsers import  get_result_path, process_data, get_relevant_path
from Modules.ConclusionsFunction.sorter import sort
from Modules.ConclusionsFunction.spliter import split_good_bad_runs
from Modules.ConclusionsFunction.calc_mean_median import calc_mean_median
from Modules.ConclusionsFunction.histogram_boxplot import plot_histograms_and_boxplots
from Modules.ConclusionsFunction.kde import plot_kde
from Modules.ConclusionsFunction.regression_heatmap import plot_regression, plot_heatmap
from Modules.ConclusionsFunction.draw_params import generate_data_base_param
from Modules.ConclusionsFunction.line_match import line_match



def run(data_path, save_size=1, return_size=1):
    
    try:
        data_path = get_relevant_path(data_path)
        data_path = data_path.replace("sample", "result")
        result_path = get_result_path(data_path)
        sort(data_path, result_path)
        df_good, df_bad = split_good_bad_runs(result_path, save_size, return_size)
        param_mean_good = calc_mean_median(df_good, df_bad, result_path)
        good_runs_param_arr = process_data(f'{result_path}/GoodRuns.csv')
        bad_runs_param_arr = process_data(f'{result_path}/BadRuns.csv')
        plot_histograms_and_boxplots(good_runs_param_arr, bad_runs_param_arr, result_path)
        plot_kde(good_runs_param_arr, bad_runs_param_arr, result_path)
        corr = plot_regression(result_path)
        plot_heatmap(corr, result_path)
        generate_data_base_param(result_path, param_mean_good)
        line_match(result_path, data_path)
        
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}. Skipping this run.")

    