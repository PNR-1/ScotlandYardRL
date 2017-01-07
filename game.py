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
        self.time_list = [0,0,0,0,0]
        self.locations = [0] * 24
        self.transport_log = [[0,0,0]] * 24
        self.starting_nodes = const.choose_starting_nodes()
        self.turn_number = 0 # Retains the turn number
        self.turn_sub_counter = 0 #Retains whose turn it is during the specific turn
                                  #0 - MRx, [1->5] for detectives iteratively
        self.complete = False
        self.detective_moves = [True,True,True,True,True] #If any detective is in a location where he does
                                                          #not have the token to move away from, it is set to False

        #For metadata and debugging, we will need more data. The following variables will store metadata
        self.detectives_transport_log = [self.transport_log] * 5
        self.detectives_location_log = [self.locations] * 5
        



    def initialize_game(self):

        self.G = g_util.make_graph()
        for i in range(5):
            self.detectives.append(d_util.initialize_detective(self.starting_nodes[i+1]))
        self.MRx = x_util.initialize_x(self.starting_nodes[0])
