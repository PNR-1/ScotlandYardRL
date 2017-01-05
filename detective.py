include networkx as nx
include numpy as np

import sys
sys.path.insert(0, '/data')
import const

starting_nodes = np.array([13,26,29,34,50,53,91,94,103,112,117,132,138,141,155,174,197,198])

def initialize_detective():
    #The state of the detective is defined by number of tokens and node position.
    detective = np.array([0,0,0,0])
    #Choose random starting point
    #Set detective[0] = node
    detective[1] = const.start_taxi
    detective[2] = const.start_bus
    detective[3] = const.start_underground
    return np.copy(detective) #Returns an explicit copy and not a pointer.

def move_detective(detective,target_node,mode):
    #detective is the list
    #target_node is the node number
    #mode is a 1-D array of length 3, [taxi,bus,underground] - 1 hot encoded

    detective[0] = target_node
    if mode[0] == 1:
        detective[1] = detective[1] - 1
    elif mode[1] == 1:
        detective[2] = detective[2] - 1
    elif mode[2] == 1:
        detective[3] = detective[3] - 1

    return detective

def valid_detective_move(detective,edge):
    #detective is the detective list
    #edge is the list that contains [current node,next node,taxi,bus,underground]
    #As the edge list is created by another function and not queried, we do not check of the validness of the edge
    #We only check if the detective has enough tokens for that move
    if edge[2] == 1 and detective[1] > 0:
        return True
    elif edge[3] == 1 and detective[2] > 0:
        return True
    elif edge[4] == 1 and detective[3] > 0:
        return True
    return False








    
