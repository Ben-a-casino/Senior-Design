import numpy as np
import pygad
import matplotlib.pyplot as plt
import os

## Prepare data
data_folder = "Toy Example"

bussing_times_file = "toy_bussing_dist.csv"
walking_times_file = "toy_walking_dist.csv"

bussing_times, walking_times = (pd.read_csv(os.path.join(data_folder, file)).to_numpy()[:,1:].astype(float)
                                for file in (bussing_distances_file, walking_distances_file))

demand_file = "toy_demand.csv"

school_demand_df = pd.read_csv(os.path.join(data_folder, demand_file), index_col = 0)

routes_file = "smaller_toy_routes.csv"
route_df = pd.read_csv(os.path.join(data_folder, routes_file), index_col=0)

"""
WARNING!

I suspect the below will need slight modification
when constructing from the actual original bus
routes. They should be lists of tract numbers,
whereas in the toy example they're lists of indices

"""

original_routes = []
for route in route_df.columns:
    stops = route_df[route].dropna().astype(int).values
    original_routes.append(stops)


## Import functions from other files
from dijkstra_evaluation import evaluate_solution
from crossover_function import spread_crossover
from MutationFunction import mutate

"""
PyGAD expects custom functions to take specfic input parameters:

    crossover <= parents, offspring_size, ga_instance
    mutation <= offspring, ga_instance
    fitness <= ga_instance, solution, solution_idx

Since our functions require additional information (e.g. 
original routes) we will define each of these functions
in a baltimore network class. This way, the functions 
will have access to the outside information they need
without having to pass this data as arguments.
"""


class Baltimore:
    WT = walking_times
    BT = bussing_times
    
    D = school_demand_df
    baseline_routes = original_routes
    
    transfer_time = 10 # minutes
    delta_max = 2

    def crossover(self, parents, offspring_size, ga_instance):
        return spread_crossover(parents, offspring_size, ga_instance)

    def mutation(self, offspring, ga_instance):
        return mutate(self.baseline_routes, offspring, self.WT, self.delta_max)

    def fitness(self, ga_instance, solution, solution_idx):
        return evaluate_solution(solution, self.D, self.BT, self.WT, self.transfer_time)[0] # the evaluation returns predecessor nodes, don't need them

# Finally, we set up the genetic algorithm using the pygad library, 
# specifying the number of generations, parents mating, solutions 
# per population, number of genes, fitness function, gene type, 
# initialization range, and mutation percentage. We then run the 
# algorithm and compute the travel time for the resulting routes.

num_children = 10
num_parents = 5
num_generations = 30

first_generation = np.array(list(map(
        [original_routes for _ in range(num_children)],
        lambda child: Baltimore.mutation(child, none)
        )))

num_genes = len(route_stops) # The number of genes corresponds to the number of stops that need to be assigned to routes

# Set up the genetic algorithm with the specified parameters. The 
# GA searches over different stop orderings (chromosomes), and 
# iteratively evolves the population of solutions to find better 
# routes based on the defined fitness function.

ga_instance = pygad.GA(
    num_generations = num_generations,              # Number of iterations to run the GA
    num_parents_mating = 3,             # Number of parents solutions to be selected for mating
    sol_per_pop = num_children,                   # Number of solutions in the population
    num_genes = num_genes,              # Number of genes in each solution
    fitness_func = Baltimore().fitness,    # Fitness function to evaluate the solutions
    crossover_type = Baltimore().crossover,
    mutation_type = Baltimore().mutation,
    gene_type = [list for _ in range(len(first_generation[0]))],                  # Gene stores floating-point values
    initial_population = first_generation,
    # mutation_percent_genes = 1          # Percentage of genes to mutate in each generation
)


# EXAMPLE USAGE
initial_objective_value = Baltimore().fitness(None, original_routes, 0)[0]

ga_instance.run()

best_solution, solution_fitness, _ = ga_instance.best_solution()

print("Before routes:", before_routes)
print("After routes:", after_routes)
print("Before cost:", initial_objective_value)
print("After cost:", solution_fitness)
print("Improvement:", solution_fitness - initial_objective_value)

# plot_routes(before_routes, stops, depot, "Initial Routes")
# plot_routes(after_routes, stops, depot, "Optimized Routes")
