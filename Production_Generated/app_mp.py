import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import Globals
import csv
import loss_functions as loss
import pandas as pd
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run GA for optimizing problem.')
parser.add_argument('--task', type=int, default=1, help='task Number')
parser.add_argument('--run_type', type=str, default='', help='run_type')
args = parser.parse_args()
# Problem Definition
problem = structure()
problem.costfunc = loss.logcosh_loss
problem.nvar = 8
problem.varmin = Globals.medium_varmin
problem.varmax = Globals.medium_varmax
problem.update_vec = [0, 1]
problem.batch_size = 32
problem.params_to_optimize = { "c1" : True,
                               "c2" : True,
                               "c3" : True,
                               "c4" : True,
                               "c5" : True,
                               "c6" : True,
                               "c7" : True,
                               "c8" : False
                              }

# GA Parameters
params = structure()
params.maxit = 300
params.npop = 50
params.beta = 1
params.pc = 2
params.mu = 0.5
params.task = args.task
params.run_type = args.run_type
run_type = params.run_type

columns=['CostFunc', 'BatchSize', 'MaxIt', 'NPop', 'Beta', 'PC', 'Mu', 'BestSol1', 'BestSol2', 'BestSol3', 'BestSol4', 'BestSol5', 'AllCost', 'AvgCost']


# Run GA
out = ga.run(problem, params)

row = {
    'CostFunc': problem.costfunc.__name__,
    'BatchSize': problem.batch_size,
    'MaxIt': params.maxit,
    'NPop': params.npop,
    'Beta': params.beta,
    'PC': params.pc,
    'Mu': params.mu,
    'BestSol1': str(out.top_5[0][0]),
    'BestSol2': str(out.top_5[0][1]),
    'BestSol3': str(out.top_5[0][2]),
    'BestSol4': str(out.top_5[0][3]),
    'BestSol5': str(out.top_5[0][4]),
    'AllCost': str(out.top_5[1]),
    'AvgCost': str(sum(out.top_5[1])/len(out.top_5[1]))
}

# Check if file exists
if not os.path.exists(f'result_{run_type}.csv'):
    # If file doesn't exist, create it and write the headers
    with open(f'result_{run_type}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)

# Append the row to the CSV file
with open(f'result_{run_type}.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(row.values())

# print(out.bestsol)

# # Results
# plt.plot(out.bestcost)
# plt.xlim(0, params.maxit)
# plt.xlabel('Iterations')
# plt.ylabel('Best Cost')
# plt.title('Genetic Algorithm (GA)')
# plt.grid(True)
# plt.show()
