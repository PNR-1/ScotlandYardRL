import networkx as nx
import numpy as np
import utilities.const as const


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
    print ('ERROR ERROR ERROR')
    print('x - ',x,'\t target_node - ',target_node,'\t mode - ',mode)



def valid_x(x,edge):
    #x is the x list
    #edge is the list that contains [current node,next node,taxi,bus,underground]
    #As the edge list is created by another function and not queried, we do not check of the validness of the edge
    #We only check if x has enough tokens for that move
    if edge[2] == 1 and x[1] > 0:
        return True
    elif edge[3] == 1 and x[2] > 0:
        return True
    elif edge[4] == 1 and x[3] > 0:
        return True
    return False
    
def choose_x_move():
    print('Current Location ',self.MRx[0], end = ' ')
    next_node = int(input('Next Node for Mr.X '))
    mode = [int(x) for x in input('Enter Mode ').split()] #Enter as 0 0 1 or 0 1 0 or 1 0 0
    return next_node,mode
