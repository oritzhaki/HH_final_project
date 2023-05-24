import numpy as np
import Globals
import random

def crossover(p1, p2, update_vec):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = [random.choice(update_vec) for _ in range(len(p1.position)-1)]  # Not considering the last element in the alpha generation

    # First perform crossover for the n-1 elements
    for i in range(len(p1.position)-1):
        c1.position[i] = alpha[i]*p1.position[i] + (1-alpha[i])*p2.position[i]
        c2.position[i] = (1-alpha[i])*p1.position[i] + alpha[i]*p2.position[i]

    # Now perform crossover for the last element (which is a list)
    last_element_alpha = [random.choice(update_vec) for _ in range(len(p1.position[-1]))]
    c1_last_element = []
    c2_last_element = []
    for i in range(len(p1.position[-1])):
        c1_val = last_element_alpha[i]*p1.position[-1][i] + (1-last_element_alpha[i])*p2.position[-1][i]
        c2_val = (1-last_element_alpha[i])*p1.position[-1][i] + last_element_alpha[i]*p2.position[-1][i]
        c1_last_element.append(c1_val)
        c2_last_element.append(c2_val)
    c1.position[-1] = c1_last_element
    c2.position[-1] = c2_last_element
    return c1, c2



def mutate(x, mu, sigma):
    sigma_n = [1,1,1,1,1,1,1,1,1,1,1,1]
    y = x.deepcopy()
    flag = np.random.rand(len(x.position)-1) <= mu  # Not considering the last element in the flag generation

    ind = np.argwhere(flag)
    for index in ind:
        sigma_val = sigma[index[0]] if isinstance(sigma, list) or isinstance(sigma, np.ndarray) else sigma
        y.position[index[0]] += sigma_val * np.random.randn() * np.random.randn() 

    # Now mutate the last element (which is a list)
    flag_last_element = np.random.rand(len(x.position[-1])) <= mu
    ind_last_element = np.argwhere(flag_last_element)
    for index in ind_last_element:
        sigma_val = sigma_n[index[0]] if isinstance(sigma_n, list) or isinstance(sigma_n, np.ndarray) else sigma
        y.position[-1][index[0]] += sigma_val * np.random.randn() * np.random.randn() 

    return y



def apply_bound(x, varmin, varmax, reset=0.5):
    for i in range(len(x.position) - 1):
        try:
            if x.position[i] < varmin[i]:
                x.position[i] = reset
            elif x.position[i] > varmax[i]:
                x.position[i] = reset
        except ValueError as e:
            print(f'Error at index {i}: {e}')
            print(f'x.position[{i}] = {x.position[i]}')
            print(f'varmin[{i}] = {varmin[i]}')
            print(f'varmax[{i}] = {varmax[i]}')
            raise

    # Apply the bounds for the list at the last position
    if isinstance(x.position[-1], list):
        for i in range(len(x.position[-1])):
            if x.position[-1][i] < varmin[-1][i]:
                x.position[-1][i] = reset
            elif x.position[-1][i] > varmax[-1][i]:
                x.position[-1][i] = reset



def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

def print_top_5(bestsol, pop, it):
        print(f"First Solution : {bestsol.position}")
        print(f"Second Solution: {pop[1].position}")
        print(f"Third  Solution: {pop[2].position}")
        print(f"Fourth Solution: {pop[3].position}")
        print(f"Fifth  Solution: {pop[4].position}")
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
            x.position[i] = getattr(Globals, key.upper())
    