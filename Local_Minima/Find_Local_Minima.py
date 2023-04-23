import HH_Equations as Equations
import loss_functions as LOSS
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')
# Alternatively, you can set the working directory to a specific path
os.chdir('C:\\Users\\galle\\OneDrive\\Desktop\\Directories\\Study\\ShanaC\\Project\\Git_HH\\Local_Minima')

C1 = 0.01
C2 = 55
C3 = 0.1
C4 = 55
C5 = 0.125
C6 = 0.0125
C7 = 65
C8 = 4

c1_range = np.arange(0, 1.0, 0.005)
c2_range = np.arange(0, 100, 0.05)
c3_range = np.arange(0, 1.0, 0.005)
c4_range = np.arange(0, 100, 0.05)
c5_range = np.arange(0, 1.0, 0.005)
c6_range = np.arange(0, 1.0, 0.005)
c7_range = np.arange(0, 100, 0.05)
c8_range = np.arange(0, 10.0, 0.05)

batchSize = 64

results = []

for i in range(len([c1_range, c2_range, c3_range, c4_range, c5_range, c6_range, c7_range, c8_range])):
    for j in range(i+1, len([c1_range, c2_range, c3_range, c4_range, c5_range, c6_range, c7_range, c8_range])):
        for param in tqdm(eval(f'c{i+1}_range'), desc=f'c{i+1}', leave=False):
            for param2 in tqdm(eval(f'c{j+1}_range'), desc=f'c{j+1}', leave=False):
                params = [C1, C2, C3, C4, C5, C6, C7, C8]
                params[i] = param
                params[j] = param2
                l2 = LOSS.l2_loss(params, batchSize)
                l1 = LOSS.l1_loss(params, batchSize)
                results.append([params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], l2, l1])

df = pd.DataFrame(results, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'L2', 'L1'])
df.to_csv('results.csv', index=False)
