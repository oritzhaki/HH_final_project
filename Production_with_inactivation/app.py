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
problem.nvar = 13
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
                               "c8" : False,
                               "c9" : True,
                               "c10": True,
                               "c11": True,
                               "c12": True,
                               "c13": True
                              }

# GA Parameters
params = structure()
params.maxit = 500
params.npop = 70
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

