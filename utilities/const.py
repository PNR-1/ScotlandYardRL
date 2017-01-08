import numpy as np

start_taxi = 10
start_bus = 8
start_underground = 4
start_taxi_x = 4
start_bus_x = 3
start_underground_x = 3

starting_nodes = np.array([13,26,29,34,50,53,91,94,103,112,117,132,138,141,155,174,197,198])

def choose_starting_nodes():
    return np.random.choice(starting_nodes, size=6, replace=False)

surface_points = np.array([2,7,12,17,23]) # Mr. X's location is revealed after turns 3,8,13,18,24
                                          # Note: Turns start as 0,1,2,3...... and the maximum turns are
                                          # (10 + 8 + 4 = ) 22. So after turn counter hits 21, game ends. 
