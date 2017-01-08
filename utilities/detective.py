import networkx as nx
import numpy as np
import utilities.const as const

def initialize_detective(n):

    detective = np.array([0,0,0,0])
    detective[0] = n
    detective[1] = const.start_taxi
    detective[2] = const.start_bus
    detective[3] = const.start_underground
    return np.copy(detective) #Returns an explicit copy and not a pointer.

def move_detective(detective,target_node,mode):
    #detective is the list
    #target_node is the node number
    #mode is a 1-D array of length 3, [taxi,bus,underground] - 1 hot encoded

    if mode[0] == 1:
        detective[1] = detective[1] - 1
        detective[0] = target_node
        return np.copy(detective)
    elif mode[1] == 1:
        detective[2] = detective[2] - 1
        detective[0] = target_node
        return np.copy(detective)
    elif mode[2] == 1:
        detective[3] = detective[3] - 1
        detective[0] = target_node
        return np.copy(detective)
    print ('ERROR ERROR ERROR')
    print('x - ',x,'\t target_node - ',target_node,'\t mode - ',mode)

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

def choose_detective_move(self,detective_id):
    print(detective_id)
    print (self.detectives[detective_id][0])

    next_node = int(input('Next Node for Detective '))
    mode = [int(x) for x in input('Enter Mode ').split()] #Enter as 0 0 1 or 0 1 0 or 1 0 0
    return next_node,mode

def isNodeEmptyForDetectives(detectives,target_node):
    #Detectives can move onto empty nodes not occupied by other detectives
    result = True
    for detective in detectives:
        if detective[0] == target_node
        result = False
        break
    return result
