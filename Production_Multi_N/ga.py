import numpy as np
import ga_functions
from ypstruct import structure
from loss_functions import loss
import warnings
import Globals as G
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run(problem, params):
    
    # Problem Information
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    update_vec = problem.update_vec
    batch_size = problem.batch_size
    params_to_optimize = problem.params_to_optimize

    # Parameters
    maxit = params.maxit
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2)
    mu = params.mu
    sigma = params.sigma
    task = params.task
    run_type = params.run_type

    # Empty Individual Template
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution Ever Found
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    
    # Initialize Population
    pop = empty_individual.repeat(npop)
    for i in range(npop):
        position = list(np.random.uniform(varmin[:-1], varmax[:-1], nvar-1)) 
        last_element = list(np.random.uniform(0, 10, G.VOLTS_NUM)) 
        position.append(last_element) 
        pop[i].position = position
        pop[i].cost = loss(pop[i].position, batch_size, costfunc, run_type)
        if pop[i].cost < bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Best Cost of Iterations
    bestcost = np.empty(maxit)
    
    # Main Loop
    for it in range(maxit):
        if it % 10 == 0:
            print(f"RUN TYPE: {run_type} TASK: {task} IT {it+1}/{maxit}")
        costs = np.array([x.cost for x in pop])
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs/avg_cost
        probs = np.exp(-beta*costs)

        popc = []
        for _ in range(nc//2):
            p1 = pop[ga_functions.roulette_wheel_selection(probs)]
            p2 = pop[ga_functions.roulette_wheel_selection(probs)]
            
            c1, c2 = ga_functions.crossover(p1, p2, update_vec)

            c1 = ga_functions.mutate(c1, mu, sigma)
            c2 = ga_functions.mutate(c2, mu, sigma)

            ga_functions.apply_bound(c1, varmin, varmax)
            ga_functions.apply_bound(c2, varmin, varmax)

            ga_functions.apply_parmas_optimization_preferences(c1, params_to_optimize)
            ga_functions.apply_parmas_optimization_preferences(c2, params_to_optimize)
            
            c1.cost = loss(c1.position, batch_size, costfunc, run_type)
            if c1.cost < bestsol.cost:
                bestsol = c1.deepcopy()

            c2.cost = loss(c2.position, batch_size, costfunc, run_type)
            if c2.cost < bestsol.cost:
                bestsol = c2.deepcopy()

            popc.append(c1)
            popc.append(c2)
        
        # Merge, Sort and Select
        pop += popc
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:npop]
        # Store Best Cost
        bestcost[it] = bestsol.cost

        # Show Iteration Information
        # ga_functions.print_top_5(bestsol, pop, it)

    # Output
    out = structure()
    out.pop = pop
    out.bestsol = bestsol
    out.top_5 = ga_functions.get_top_5(bestsol, pop)
    out.bestcost = bestcost
    return out

