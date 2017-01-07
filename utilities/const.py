import numpy as np

start_taxi = 10
start_bus = 8
start_underground = 4
start_taxi_x = 1
start_bus_x = 1
start_underground_x = 1

starting_nodes = np.array([13,26,29,34,50,53,91,94,103,112,117,132,138,141,155,174,197,198])

def choose_starting_nodes():
    return np.random.choice(starting_nodes, size=6, replace=False)
