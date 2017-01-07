import numpy as np
import networkx as nx
import utilities.graph_utils as g_util #Graph utilities
import utilities.detective as d_util #detective's utilities
import utilities.mrx as x_util #MR X's utilities
import utilities.const as const

class ScotlandYard(object):
    def __init__(self):
        self.G = []
        self.detectives = []
        self.MRx = []
        self.prev_5 = [] #Notes down the last 5 transactions by MRx
        self.last_location = [] #The last location of MRx
        self.time_list = [0,0,0,0,0]
        self.starting_nodes = const.choose_starting_nodes()


    def initialize_game(self):

        self.G = g_util.make_graph()
        for i in range(5):
            self.detectives.append(d_util.initialize_detective(self.starting_nodes[i+1]))
        self.MRx = x_util.initialize_x(self.starting_nodes[0])
        self.prev_5 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
        self.last_location = 0
        
