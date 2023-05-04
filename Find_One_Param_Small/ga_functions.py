import numpy as np
import Globals
import random

def crossover(p1, p2, update_vec):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = [random.choice(update_vec) for _ in range(len(p1.position))]
    for i in range(len(p1.position)):
        c1.position[i] = alpha[i]*p1.position[i] + (1-alpha[i])*p2.position[i]
        c2.position[i] = (1-alpha[i])*p1.position[i] + alpha[i]*p2.position[i]
    return c1, c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma * np.random.randn(*ind.shape) * np.random.randn(*ind.shape)
    return y

def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

def print_top_5(bestsol, pop, it):
        tranformation_vector = [0.1, 100, 1, 100, 1, 0.1, 100, 1]
        print(f"First Solution : {[x*y for x, y in zip(pop[0].position, tranformation_vector)]}")
        print(f"Second Solution : {[x*y for x, y in zip(pop[1].position, tranformation_vector)]}")
        print(f"Third Solution : {[x*y for x, y in zip(pop[2].position, tranformation_vector)]}")
        print(f"Fourth Solution : {[x*y for x, y in zip(pop[3].position, tranformation_vector)]}")
        print(f"Fifth Solution : {[x*y for x, y in zip(pop[4].position, tranformation_vector)]}")
        print("Iteration {}: First  Cost = {}".format(it, bestsol.cost))
        print("Iteration {}: Second Cost = {}".format(it, pop[1].cost))
        print("Iteration {}: Third  Cost = {}".format(it, pop[2].cost))
        print("Iteration {}: Fourth Cost = {}".format(it, pop[3].cost))
        print("Iteration {}: Fifth  Cost = {}".format(it, pop[4].cost))
        
def get_top_5(bestsol, pop):
    top_5 = [bestsol] + pop[1:5]
    return [sol.position for sol in top_5], [sol.cost for sol in top_5]

def apply_parmas_optimization_preferences(x, preferences):
    for i, (key, optimize) in enumerate(preferences.items()):
        if not optimize:
            if key == "c8":
                x.position[i] = getattr(Globals, key.upper())
            else:
                if x.position[i] > 0:
                    x.position[i] = getattr(Globals, key.upper()) / 2
                else:
                    x.position[i] = getattr(Globals, key.upper()) * 2