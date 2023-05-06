import numpy as np
import HH_Equations as Equations
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import Globals

def tranform(vector):
    tranformed_vector_for_loss_calc = [x*y for x, y in zip(vector, Globals.tranformation_vector)]
    return tranformed_vector_for_loss_calc

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

t, V, labels = get_data('../GeneratedData/OldApproach/dataset.csv')


def get_batch_indices(batchSize):
    indices = []
    for i in range(0, len(t), 100):
        indices += list(range(i, i+20))
    
    remaining_indices = set(range(len(t))) - set(indices)
    random_indices = random.sample(remaining_indices, batchSize)
    indices += random_indices
    return indices

def l2_loss(params, batchSize):
    loss = 0
    indices = get_batch_indices(batchSize)
    for i in indices:
        y_hat = np.round(Equations.get_y_hat(tranform(params), t[i], V[i]), 8)
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1,2) * 100
        loss += (np.round(y_hat,8) - np.round(labels[i],8)) ** 2
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1,2) * 1000
    ret_loss = loss / len(indices)
    return ret_loss


def l1_loss(params, batchSize):
    loss = 0
    indices = get_batch_indices(batchSize)
    for i in indices:
        y_hat = np.round(Equations.get_y_hat(tranform(params), t[i], V[i]), 8)
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1,2) * 100
        loss += np.abs(y_hat - labels[i])
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1,2) * 1000
    ret_loss = loss / len(indices)
    return ret_loss

def logcosh_loss(params, batchSize):
    loss = 0
    indices = get_batch_indices(batchSize)
    for i in indices:
        y_hat = np.round(Equations.get_y_hat(tranform(params), t[i], V[i]), 8)
        if y_hat == float("inf") or np.isnan(y_hat):
            y_hat = np.random.uniform(1, 2) * 100
        loss += np.log(np.cosh(y_hat - labels[i]))
    if loss == float("inf") or np.isnan(loss): loss = np.random.uniform(1, 2) * 1000
    ret_loss = loss / len(indices)  # the average of all t, V's checked
    return ret_loss