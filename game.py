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
        self.time_list = np.array([0,0,0,0,0])
        self.locations = np.array([0] * 24)   #For MRx
        self.transport_log = np.array([[0,0,0]] * 24)   #For MRx
        self.starting_nodes = const.choose_starting_nodes()
        self.turn_number = 0 # Retains the turn number that has been completed
        self.turn_sub_counter = 0 #Retains whose turn it is during the specific turn
                                  #0 - MRx, [1->5] for detectives iteratively
        self.complete = False
        self.detective_moves = [True,True,True,True,True] #If any detective is in a location where he does
                                                          #not have the token to move away from, it is set to False

        #For metadata and debugging, we will need more data. The following variables will store metadata
        self.detectives_transport_log = np.array([self.transport_log] * 5)
        self.detectives_location_log = np.array([self.locations] * 5)

    def initialize_game(self):

        self.G = g_util.make_graph()
        for i in range(5):
            self.detectives.append(d_util.initialize_detective(self.starting_nodes[i+1]))
        self.MRx = x_util.initialize_x(self.starting_nodes[0])

    def take_action(self,next_node,mode):
        if self.complete == True:
            print ('Game Over!')
            return
        if self.turn_sub_counter == 0:
            self.MRx = x_util.move_x(self.MRx,next_node,mode) #Can pass extra arguments
            self.transport_log[self.turn_number] = mode
            self.locations[self.turn_number] = next_node
            self.turn_sub_counter = self.turn_sub_counter + 1
        else:
            if self.detective_moves[self.turn_sub_counter - 1] == False:
                return
            self.detectives[self.turn_sub_counter - 1] = d_util.move_detective(self.detectives[self.turn_sub_counter - 1],next_node,mode)
            self.detectives_transport_log[self.turn_sub_counter - 1][self.turn_number] = mode
            self.detectives_location_log[self.turn_sub_counter - 1][self.turn_number] = next_node
            self.turn_sub_counter = self.turn_sub_counter + 1

        if self.turn_sub_counter == 6: #Turn over, reset turn_number
            self.turn_number = self.turn_number + 1
            self.turn_sub_counter = 0
        #self.update(); #Checks if game has ended

    def update(self):
        assertion1 = False #Assume no detective is on MRx's node
        for detective in self.detectives:
            if detective[0] == self.MRx[0]:
                assertion1 = True
                break
        assertion2 = False #Assume no detective has any legal moves left
        #Check if detective has no legal moves
        for i in range(5):
            if self.detective_moves[i] == True:
                self.detective_moves[i] = d_util.detective_has_valid_moves()
            if self.detective_moves[i] == True:
                assertion2 = True

        if assertion1 == True or assertion2 == False:
            self.complete = True
