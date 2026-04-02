import os

# Implementation of Dijkstra's algorithm
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra

# Do other math
import pandas as pd
import numpy as np

# Visualization
import networkx as nx
import matplotlib.pyplot as plt



def construct_adjacency_matrix(route_df, WD, BD, transfer_time):
    # route_df = DataFrame containing the order of census tracts each bus route visits
        # Column = Bus route, row = stop order
    # WD = matrix of walking distances
    # BD = matrix of bussing distances
    # transfer_time = amount of time spent waiting for a bus, on average
    
    # Start building the adjacency matrix from the walking distances
    A = WD
    
    transfer_node_index = len(WD)
    
    for route in route_df.columns:

        stops = route_df[route].dropna().astype(int).values
        
        # Increase the size of the adjacency matrix to handle extra transfer nodes
        A = np.hstack((A,np.zeros((A.shape[0],len(stops)))))
        A = np.vstack((A,np.zeros((len(stops),A.shape[1]))))

        for i,stop in enumerate(stops):
            j = transfer_node_index + i

            # Waiting to transfer onto the bus takes time
            A[stop,j] = transfer_time

            # Getting off the bus is negligible
            A[j,stop] = 0.001

        # Connect the nodes on the bus route based on bus distance matrix
        for i in range(len(stops)-1):
            j = transfer_node_index + i
            stop0 = stops[i]
            stop1 = stops[i+1]

            A[j,j+1] = BD[stop0,stop1]
            A[j+1,j] = BD[stop1,stop0]

        # Update the index for the next set of transfer nodes
        transfer_node_index += len(stops)
    return A

## Run dijkstras
def evaluate_solution(route_df, school_demand_df, BD, WD, transfer_time):
    # route_df = DataFrame of routes, where each column lists the tracts that bus route visits
    # school_demand_df = DataFrame where element D[i,j] = # of students from tract i going to tract j
    # WD = matrix of walking distances
    # BD = matrix of bussing distances
    # transfer_time = amount of time spent waiting for the next bus

    A = construct_adjacency_matrix(route_df, WD, BD, transfer_time)
 
    T, predecessors = dijkstra(csr_array(A), directed = True, return_predecessors = True)
    
    # T = time matrix where T[i,j] = minimum amount of time needed to travel from tract i to tract j
    # predecessors[i,j] = the node before reaching j on the shortest path from i to j
        # EX: predecessors[0,3] = 2 ==> The computer reached node 3 from node 2 on the shortest path from 0 to 3

    D = school_demand_df.to_numpy()

    # Wherever there's demand, multiply by the time spent travelling
    C = T[:D.shape[0], :D.shape[1]] * D 
    
    # Square the cost to disencentivize longer commutes for more people
    return np.sum(C**2), predecessors 

if __name__ == "__main__":

    data_folder = "Toy Example"
    bussing_distances_file = "toy_bussing_dist.csv"
    walking_distances_file = "toy_walking_dist.csv"
    demand_file = "toy_demand.csv"

    BD, WD = (pd.read_csv(os.path.join(data_folder, file)).to_numpy()[:,1:].astype(float)
              for file in (bussing_distances_file, walking_distances_file))

    school_demand_df = pd.read_csv(os.path.join(data_folder, demand_file), index_col = 0)
    
    transfer_time = 10 # minutes

    routes_file = "smaller_toy_routes.csv"
    route_df = pd.read_csv(os.path.join(routes_file), index_col=0)

    total_cost, predecessors = evaluate_solution(route_df, school_demand_df, BD, WD, transfer_time) 

    ## Visualize results
    
    """
    The coordinates and colors are specific to a 3x3 grid.
    If you want to use a different example,
    choose a new set of coordinates to represent the 
    problem. NetworkX provides the "spring_layout"
    function for this purpose.
    """

    coordinates = {}
    colors = {}
    i = 0
    for y in range(3):
        for x in range(3):
            coordinates[i] = [x,y,0]
            colors[i] = "black"
            i += 1

    for j,route in enumerate(route_df.columns):
        stops = route_df[route].dropna().astype(int).values
        for stop in stops:
            coordinates[i] = [*coordinates[stop][:-1],j+1]
            colors[i] = route.lower()
            i += 1

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for key in coordinates.keys():
        ax.scatter(*coordinates[key], color = colors[key])

    # Plot the shortest path
    for row in range(school_demand_df.shape[0]):
        for col in range(school_demand_df.shape[1]):
            # Plot the trip each time there's demand
            if school_demand_df.iloc[row, col] > 0:
                start = row
                target = col
                priors = predecessors[start, :]

                # Walk "backwards" through the previous tracts until you reach the starting node
                while True:
                    previous = priors[target]

                    # Draw the line between the two tracts, and label it with the demand it corresponds to
                    x0,y0,z0 = coordinates[previous]
                    x1,y1,z1 = coordinates[target]
                    ax.plot([x0,x1], [y0,y1], [z0,z1], color = "black", linestyle = "dashed")
                    ax.text(*(sum(c) / 2 for c in ([x0,x1], [y0,y1], [z0,z1])), f"{row,col}", ha = "center", va = "center")

                    target = previous

                    if previous == start:
                        break

    plt.show()

