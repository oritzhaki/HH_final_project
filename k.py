import numpy as np

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

beta = 1
costs = np.array([0.05, 0.1, 0.2])
avg_cost = np.mean(costs)
if avg_cost != 0:
    costs = costs/avg_cost
probs = np.exp(-beta*costs)

roulette_wheel_selection(probs)