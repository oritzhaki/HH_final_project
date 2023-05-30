import numpy as np
import HH_Equations as Equations
import random
import pandas as pd
import os
import Globals as G
from functools import lru_cache



def count_csv_files():
    directory = f'ProdDataRats/{G.CURRENT_CELL}'
    csv_count = len([filename for filename in os.listdir(directory) if filename.endswith('.csv')])
    return csv_count

CSV_FILES_NUM = count_csv_files()
TRAIN = int(0.8 * CSV_FILES_NUM)
TEST = CSV_FILES_NUM - TRAIN

@lru_cache(maxsize=None)  # Cache the result of get_data to avoid unnecessary I/O operations
def get_data(path):
    data = pd.read_csv(path)
    t = data.iloc[:, 0].values
    V = data.iloc[:, 1].values
    labels = data.iloc[:, 2].values
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
    return (y_hat - y) ** 2

def logcosh_loss(y_hat, y):
    return np.log(np.cosh(y_hat - y))

def calc_loss_onetime(params, batchSize, costfunc, i):
    t, V, labels = get_data(f'ProdDataRats/ProdDataRatsRemoved/{G.CURRENT_CELL}/sample_{i}_removed.csv')
    indices = np.array(get_batch_indices(batchSize, t))
    t_batch = t[indices]
    V_batch = V[indices]
    y_hat = np.round(Equations.get_y_hat(params, t_batch, V_batch), 8)
    y_hat = np.where(np.isinf(y_hat) | np.isnan(y_hat), np.random.uniform(1,2) * 100, y_hat)
    loss = costfunc(y_hat, labels[indices])
    loss = np.where(np.isinf(loss) | np.isnan(loss), np.random.uniform(1,2) * 1000, loss)
    ret_loss = loss.mean()
    return ret_loss

def loss(params, batchSize, costfunc, run_type):
    if run_type == 'train':
        return np.mean([calc_loss_onetime(params, batchSize, costfunc, i) for i in range(1, TRAIN + 1)])
    if run_type == 'test':
        return np.mean([calc_loss_onetime(params, batchSize, costfunc, i) for i in range(CSV_FILES_NUM - TEST + 1, CSV_FILES_NUM + 1)])

