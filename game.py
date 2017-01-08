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

    def play_turn(self):

        if self.turn_sub_counter == 0: #Redundant Statement
            self.play_MRx()
            self.turn_sub_counter = self.turn_sub_counter + 1

        for i in range(5):
            if self.detective_moves[i] == True:
                self.play_detective(i) #Passing 0,1,2,3,4
            self.turn_sub_counter = self.turn_sub_counter + 1 #Goes 1,2,3,4,5

        self.turn_number = self.turn_number + 1
        self.turn_sub_counter = 0
        #isGameEnd() Implement This function

    def play_MRx(self):
        next_node = 0
        mode = [0,0,0]

        next_node,mode = self.choose_x_move()
        mode = np.array(mode)
        print(next_node,mode)
        self.MRx = x_util.move_x(self.MRx,next_node,mode)
        self.transport_log[self.turn_number] = mode
        self.locations[self.turn_number] = next_node
        del next_node,mode


    def play_detective(self,detective_id):
        next_node = 0
        mode = [0,0,0]

        next_node,mode = self.choose_detective_move(detective_id)
        print(next_node,mode)
        self.detectives[detective_id] = d_util.move_detective(self.detectives[detective_id],next_node,mode)
        self.detectives_transport_log[detective_id][self.turn_number] = mode
        self.detectives_location_log[detective_id][self.turn_number] = next_node
        del next_node,mode

    #Function to be changed to use RL methods
    def choose_x_move(self):
        print('Current Location ',self.MRx[0], end = ' ')
        next_node = int(input('Next Node for Mr.X '))
        mode = [int(x) for x in input('Enter Mode ').split()] #Enter as 0 0 1 or 0 1 0 or 1 0 0
        return next_node,mode

    #Function to be changed to use RL methods
    def choose_detective_move(self,detective_id):
        print(detective_id)
        print (self.detectives[detective_id][0])

        next_node = int(input('Next Node for Detective '))
        mode = [int(x) for x in input('Enter Mode ').split()] #Enter as 0 0 1 or 0 1 0 or 1 0 0
        return next_node,mode
