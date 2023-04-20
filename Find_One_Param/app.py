
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import ga
import Globals
import loss_functions as loss
import pandas as pd

# Problem Definition
problem = structure()
problem.costfunc = loss.l2_loss
problem.nvar = 8
problem.varmin = Globals.medium_varmin
problem.varmax = Globals.medium_varmax
problem.update_vec = Globals.medium_gamma
problem.batch_size = 64
problem.params_to_optimize = { "c1" : True,
                               "c2" : True,
                               "c3" : True,
                               "c4" : True,
                               "c5" : False,
                               "c6" : False,
                               "c7" : False,
                               "c8" : False
                              }

# GA Parameters
params = structure()
params.maxit = 1000
params.npop = 100
params.beta = 1
params.pc = 2
params.mu = 0.2
params.sigma = Globals.medium_sigma

# Run GA
out = ga.run(problem, params)
print(out.bestsol)

# Results
plt.plot(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iterations')
plt.ylabel('Best Cost')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.show()

