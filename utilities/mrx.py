import networkx as nx
import numpy as np
import utilities.const as const
import utilities.graph_utils as g_util


def initialize_x(n):
    #The state of the Mr X is defined by number of tokens and node position.
    x = np.array([0,0,0,0])
    x[0] = n
    x[1] = const.start_taxi_x
    x[2] = const.start_bus_x
    x[3] = const.start_underground_x
    return np.copy(x) #Returns an explicit copy and not a pointer.

def move_x(x,target_node,mode):
    #x is the list
    #target_node is the node number
    #mode is a 1-D array of length 3, [taxi,bus,underground] - 1 hot encoded

    if mode[0] == 1:
        x[1] = x[1] - 1
        x[0] = target_node
        return np.copy(x)
    elif mode[1] == 1:
        x[2] = x[2] - 1
        x[0] = target_node
        return np.copy(x)
    elif mode[2] == 1:
        x[3] = x[3] - 1
        x[0] = target_node
        return np.copy(x)
    return np.copy(x)

def x_valid_list(MRx,G):

    current_node = MRx[0]
    taxi = MRx[1]
    bus = MRx[2]
    underground = MRx[3]

    edges = g_util.connections(G,current_node)

    x_valid_list = []

    for edge in edges:
        if edge[2] == 1 and taxi > 0:
            x_valid_list.append(edge)
        elif edge[3] == 1 and bus > 0:
            x_valid_list.append(edge)
        elif edge[4] == 1 and underground > 0:
            x_valid_list.append(edge)

    return np.array(x_valid_list)
