import HH_Equations as Equations
import loss_functions as LOSS
import numpy as np
import pandas as pd
from tqdm import tqdm

c1_range = np.arange(0, 1.0, 0.01)
c2_range = np.arange(0, 100, 0.1)
c3_range = np.arange(0, 1.0, 0.01)
c4_range = np.arange(0, 100, 0.1)
c5_range = np.arange(0, 1.0, 0.01)
c6_range = np.arange(0, 1.0, 0.01)
c7_range = np.arange(0, 100, 0.1)
c8 = 4

batchSize = 1

results = []

for c1 in tqdm(c1_range, desc='c1'):
    for c2 in tqdm(c2_range, desc='c2', leave=False):
        for c3 in tqdm(c3_range, desc='c3', leave=False):
            for c4 in tqdm(c4_range, desc='c4', leave=False):
                for c5 in tqdm(c5_range, desc='c5', leave=False):
                    for c6 in tqdm(c6_range, desc='c6', leave=False):
                        for c7 in c7_range:
                            params = [c1, c2, c3, c4, c5, c6, c7, 4]
                            l2 = LOSS.l2_loss(params, batchSize)
                            l1 = LOSS.l1_loss(params, batchSize)
                            results.append([c1, c2, c3, c4, c5, c6, c7, c8, l2, l1])
                            # print(f"c1={c1}, c2={c2}, c3={c3}, c4={c4}, c5={c5}, c6={c6}, c7={c7}, c8={c8}, L2={l2:.6f}, L1={l1:.6f}")


df = pd.DataFrame(results, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'L2', 'L1'])
df.to_csv('results.csv', index=False)
