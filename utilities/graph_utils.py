import pandas as pd
import networkx as nx
import numpy as np

def make_graph():
    taxi = pd.read_csv('data/taxi.csv')
    bus = pd.read_csv('data/bus.csv')
    underground = pd.read_csv('data/underground.csv')

    G = nx.MultiGraph()
    G.add_nodes_from( range(1,200) )
    for i in range(len(taxi)):
        G.add_edge( taxi['Node1'][i] , taxi['Node2'][i] , type = 'taxi' , color = 'yellow' )
    for i in range(len(bus)):
        G.add_edge( bus['Node1'][i] , bus['Node2'][i] , type = 'bus' , color = 'blue' )
    for i in range(len(underground)):
        G.add_edge( underground['Node1'][i] , underground['Node2'][i] , type = 'underground' , color = 'red' )

    return G

#Takes graph G and node number node.
#Returns a list of connections
#Format Node 1,Node 2, taxi, bus, underground
def connections(G,node):
    all_connections = G.edges([node],data=True)
    list = []
    for i in range(len(all_connections)):
        node1,node2,data = all_connections[i]
        add_list = [0,0,0,0,0]
        add_list[0] = node1
        add_list[1] = node2
        if (data['type'] == 'taxi'):
            add_list[2] = 1
        elif (data['type'] == 'bus'):
            add_list[3] = 1
        elif (data['type'] == 'underground'):
            add_list[4] = 1
        list.append(add_list)
    return np.copy(list)

def node_one_hot(n):
    list = [0] * 199
    list[n-1] = 1 #Cause nodes are 1,2,3,4,5,6,7....199 and list is 0,1,....,198
    return list

def node_one_hot_zero():
    list = [0] * 199
    return list

def print_list(D_3_list): 
    number_of_det,rows,_ = D_3_list.shape
    String = ""
    for row in range(rows):
        for det in range(number_of_det):
            String=String+(str(D_3_list[det][row]))+'\t'
        String=String + '\n'
    return String
