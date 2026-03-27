import numpy as np
import pygad
import matplotlib.pyplot as plt


# This function plots the bus routes on a 2D plane, showing the stops and the depot.
def plot_routes(routes, stops, depot, title):
    plt.figure(figsize=(8, 6))

    # Plot stops
    for stop_id, (x, y) in stops.items():
        if stop_id == depot:
            plt.scatter(x, y, s = 180, label='Depot/School')
            plt.text(x + 0.05, y + 0.05, f'Depot {stop_id}', fontsize=9)
        else:
            plt.scatter(x, y, s = 50, c = 'black')
            plt.text(x + 0.05, y + 0.05, f'Stop {stop_id}', fontsize=9)

    # Plot each bus route
    for b, route in enumerate(routes):
        if not route:
            continue
        
        # Add the depot at the start and end of the route
        full_route = [depot] + route + [depot]

        # Loop through the full route and plot arrows between consecutive stops
        for j in range(len(full_route) - 1):
            start = full_route[j]       # Get the current stop
            end = full_route[j + 1]     # Get the next stop

            x1, y1 = stops[start]       # Get the coordinates of the current stop
            x2, y2 = stops[end]         # Get the coordinates of the next stop

            dx = x2 - x1                # Calculate the change in x and y for the arrow
            dy = y2 - y1

            # Plot the arrow representing the bus route between the current stop and the next stop
            plt.arrow(
                x1, y1, 
                dx, dy, 
                head_width=0.1,
                head_length=0.1, 
                length_includes_head=True, 
                fc = f"C{b}", ec = f"C{b}",
                alpha = 0.8)

    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# This is a simplified capacity-based route assignment example and not yet fully constrained.
# We begin by defining the problem parameters, including the stops, 
# depot, route stops, number of buses, bus capacities, student counts 
# at each stop, and the start and end nodes for each bus.

stops = {
    0: (0, 0),
    1: (1, 2),
    2: (2, 1),
    3: (3, 3),
    4: (4, 0)
}

depot = 0
route_stops = [stop for stop in stops.keys() if stop != depot]

num_buses = 2

bus_cap = [5, 5]

student_count = {
    1: 2,
    2: 2,
    3: 1,
    4: 2
}

# Next, we compute the travel time between each pair of stops, 
# specifically using the Euclidean distance formula for this example.

def compute_travel_time(stops):
    n = len(stops)
    travel_time = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
                x1, y1 = stops[i]
                x2, y2 = stops[j]
                travel_time[i][j] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return travel_time

travel_time = compute_travel_time(stops)

# The decode_solution function takes a solution from the genetic 
# algorithm and decodes it into a set of routes for the buses, 
# ensuring that the capacity constraints are respected. However,
# in a real-world implementation, additional logic would be needed to handle
# cases where capacity or bus availability constraints are violated.

def decode_solution(solution):
    order = [x for _, x in sorted(zip(solution, route_stops))] # Sort the route stops based on the solution values

    routes = []
    current_route = []
    current_load = 0
    bus_idx = 0

    # Iterate through the ordered stops and assign them to routes while respecting bus capacities
    for stop in order:
        demand = student_count[stop] # Get the number of students at the current stop

        if bus_idx < num_buses and current_load + demand <= bus_cap[bus_idx]:
            # If the current bus can accommodate the demand of the stop, add it to the current route and update the load
            current_route.append(stop)
            current_load += demand
        
        else:
            # If the current bus cannot accommodate the demand, finalize the current route and start a new one for the next bus
            routes.append(current_route)
            bus_idx += 1
            current_route = [stop]
            current_load = demand

    if current_route:
        routes.append(current_route) # Add the last route if it has any stops

    return routes

# The route_cost function calculates the total cost of a set of routes,
# which is the sum of the travel times for all routes, including the 
# return to the depot. It iterates through each route to construct the 
# full route and sums the travel times between consecutive stops.

def route_cost(routes, stops, depot):
    total_cost = 0 

    for route in routes:
        if not route:
            continue

        full_route = [depot] + route + [depot] # Construct the full route by adding the depot at the start and end of the route

        # Iterate through the full route and sum the travel times between consecutive stops
        for i in range(len(full_route) - 1):
            total_cost += travel_time[full_route[i]][full_route[i + 1]] # Add the travel time between the current stop and the next stop to the total cost

    return total_cost

# The fitness_function evaluates the fitness of a solution by decoding it
# into routes, calculating the total cost of those routes, and then 
# returning a fitness value that is inversely related to the cost.

def fitness_function(ga_instance, solution, solution_idx):
    routes = decode_solution(solution) # Decode the solution into routes for the buses
    cost = route_cost(routes, stops, depot) # Calculate the total cost of the routes based on the travel times between stops

    return  1.0 / (1.0 + cost)


# Finally, we set up the genetic algorithm using the pygad library, 
# specifying the number of generations, parents mating, solutions 
# per population, number of genes, fitness function, gene type, 
# initialization range, and mutation percentage. We then run the 
# algorithm and compute the travel time for the resulting routes.

num_genes = len(route_stops) # The number of genes corresponds to the number of stops that need to be assigned to routes

# Set up the genetic algorithm with the specified parameters. The 
# GA searches over different stop orderings (chromosomes), and 
# iteratively evolves the population of solutions to find better 
# routes based on the defined fitness function.

ga_instance = pygad.GA(
    num_generations = 100,              # Number of iterations to run the GA
    num_parents_mating = 4,             # Number of parents solutions to be selected for mating
    sol_per_pop = 10,                   # Number of solutions in the population
    num_genes = num_genes,              # Number of genes in each solution
    fitness_func = fitness_function,    # Fitness function to evaluate the solutions
    gene_type = float,                  # Gene stores floating-point values
    init_range_low = 0.0,               # Lower bound for initializing gene values
    init_range_high = 1.0,              # Upper bound for initializing gene values
    mutation_percent_genes = 1          # Percentage of genes to mutate in each generation
)


# EXAMPLE USAGE


baseline_solution = [0.1, 0.2, 0.3, 0.4]
before_routes = decode_solution(baseline_solution)

ga_instance.run()

best_solution, solution_fitness, _ = ga_instance.best_solution()
after_routes = decode_solution(best_solution)

print("Before routes:", before_routes)
print("After routes:", after_routes)
print("Before cost:", route_cost(before_routes, stops, depot))
print("After cost:", route_cost(after_routes, stops, depot))
print("Improvement:", route_cost(before_routes, stops, depot) - route_cost(after_routes, stops, depot))

plot_routes(before_routes, stops, depot, "Initial Routes")
plot_routes(after_routes, stops, depot, "Optimized Routes")
