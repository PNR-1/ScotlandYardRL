import networkx as nx
import numpy as np
import utilities.const as const
import utilities.graph_utils as g_util


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

def dec_valid_list(detectives,G,detective_id):
    #detective_id is the id of detective - [0,4]

    current_node = detectives[detective_id][0]
    taxi = detectives[detective_id][1]
    bus = detectives[detective_id][2]
    underground = detectives[detective_id][3]

    edges = g_util.connections(G,current_node)

    x_valid_list = []

    for edge in edges:
        if edge[2] == 1 and taxi > 0:
            x_valid_list.append(edge)
        elif edge[3] == 1 and bus > 0:
            x_valid_list.append(edge)
        elif edge[4] == 1 and underground > 0:
            x_valid_list.append(edge)

    valid_list = []

    for edge in x_valid_list:
        invalid_move = False
        for detective in detectives:
            if detective[0] == edge[1]:
                invalid_move = True
                break
        if invalid_move == False:
            valid_list.append(edge)

    return np.array(valid_list)

def send_token(MRx,mode):
    if mode[0] == 1:
        MRx[1] = MRx[1] + 1
    elif mode[1] == 1:
        MRx[2] = MRx[2] + 1
    elif mode[3] == 1:
        MRx[3] = MRx[3] + 1

    return MRx 
