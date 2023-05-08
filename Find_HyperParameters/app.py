
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import Globals
import loss_functions as loss
import pandas as pd
import csv
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

costfuncs = [loss.logcosh_loss, loss.l2_loss, loss.l1_loss]
batch_sizes = [32]
maxits = [500]
npops = [100]
betas = [0, 1, 10]
pcs = [2]
mus = [0.4, 0.8]
sigmas = [Globals.extreme_sigma]


columns=['CostFunc', 'BatchSize', 'MaxIt', 'NPop', 'Beta', 'PC', 'Mu', 'Sigma', 'BestSol1', 'BestSol2', 'BestSol3', 'BestSol4', 'BestSol5', 'AllCost', 'AvgCost']


num_combinations = len(costfuncs) * len(batch_sizes) * len(maxits) * len(npops) * len(betas) * len(pcs) * len(mus) * len(sigmas)
counter = 1

for costfunc in costfuncs:
    for batch_size in batch_sizes:
        for maxit in maxits:
            for npop in npops:
                for beta in betas:
                    for pc in pcs:
                        for mu in mus:
                            for sigma in sigmas:
                                
                                print(f"{counter}/{num_combinations}")
                                
                                problem = structure()
                                problem.costfunc = costfunc
                                problem.nvar = 8
                                problem.varmin = Globals.medium_varmin
                                problem.varmax = Globals.medium_varmax
                                problem.update_vec = Globals.easy_gamma
                                problem.batch_size = batch_size

                                params = structure()
                                params.maxit = maxit
                                params.npop = npop
                                params.beta = beta
                                params.pc = pc
                                params.mu = mu
                                params.sigma = sigma

                                out = ga.run(problem, params)
                                
                                row = {
                                    'CostFunc': costfunc.__name__,
                                    'BatchSize': batch_size,
                                    'MaxIt': maxit,
                                    'NPop': npop,
                                    'Beta': beta,
                                    'PC': pc,
                                    'Mu': mu,
                                    'Sigma': str(sigma),
                                    'BestSol1': str(out.top_5[0][0]),
                                    'BestSol2': str(out.top_5[0][1]),
                                    'BestSol3': str(out.top_5[0][2]),
                                    'BestSol4': str(out.top_5[0][3]),
                                    'BestSol5': str(out.top_5[0][4]),
                                    'AllCost': str(out.top_5[1]),
                                    'AvgCost': str(sum(out.top_5[1])/len(out.top_5[1]))
                                }
                                if not os.path.exists('my_dataframe.csv'):
                                    # If file doesn't exist, create it and write the headers
                                    with open('my_dataframe.csv', 'w', newline='') as f:
                                        writer = csv.writer(f)
                                        writer.writerow(columns)

                                # Append the row to the CSV file
                                with open('my_dataframe.csv', 'a', newline='') as f:
                                    writer = csv.writer(f)
                                    writer.writerow(row.values())
                                
                                counter+=1



