import numpy as np
import HH_Equations as Equations
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd

import os

def count_csv_files():
    # Directory path
    directory = 'ProdDataRats/'

    # Initialize a counter
    csv_count = 0

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            csv_count += 1

    return csv_count

CSV_FILES_NUM = count_csv_files()
TRAIN = int(0.8 * CSV_FILES_NUM)
TEST = int(0.2 * CSV_FILES_NUM)


def get_data(path):
    data = pd.read_csv(path)
    inputs = data.iloc[:, :-1]
    t, V = inputs
    t = inputs.iloc[:,0].values
    V = inputs.iloc[:,1].values
    labels = data.iloc[:,2].values
    return t, V, labels

def get_scaled_data(path):
    data = pd.read_csv(path)
    data = data.values
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    inputs = data_standardized[:, :-1]
    labels = data_standardized[:, -1]
    t = inputs[:, :-1]
    t = np.arange(t.shape[0]) 
    V = inputs[:, -1]
    return t, V, labels


def get_batch_indices(batchSize , t):
    indices = []
    for i in range(0, len(t), 100):
        indices += list(range(i, i+20))
    
    remaining_indices = set(range(len(t))) - set(indices)
    random_indices = random.sample(remaining_indices, batchSize)
    indices += random_indices
    return indices


def l1_loss(y_hat, y):
    return np.abs(y_hat - y)
def l2_loss(y_hat, y):
    return (np.round(y_hat,8) - np.round(y,8)) ** 2
def logcosh_loss(y_hat, y):
    return np.log(np.cosh(y_hat - y))

def calc_loss_onetime(params, batchSize, costfunc, i):
    t, V, labels = get_data(f'../GeneratedData/Data/Noise_Data/dataset_noise_{i}.csv')
    loss = 0
    indices = get_batch_indices(batchSize, t)
    for i in indices:
        y_hat = np.round(Equations.get_y_hat(params, t[i], V[i]), 8)
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1,2) * 100
        loss += costfunc(y_hat, labels[i])
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1,2) * 1000
    ret_loss = loss / len(indices)
    return ret_loss

def loss(params, batchSize, costfunc, run_type):
    if run_type == 'train':
        it_loss_train = 0
        for i in range(1, TRAIN + 1):
            it_loss_train += calc_loss_onetime(params, batchSize, costfunc, i)
        return it_loss_train / TRAIN
    
    if run_type == 'test':
        it_loss_train = 0
        for i in range(CSV_FILES_NUM - TEST + 1, CSV_FILES_NUM + 1):
            it_loss_train += calc_loss_onetime(params, batchSize, costfunc, i)
        return it_loss_train / TEST