import numpy as np
import networkx as nx
import utilities.graph_utils as g_util #Graph utilities
import utilities.detective as d_util #detective's utilities
import utilities.mrx as x_util #MR X's utilities
import utilities.const as const

class ScotlandYard(object):
    def __init__(self):
        self.G = []
        self.detectives = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
        self.MRx = np.array([])
        self.locations = np.array([0] * 24)   #For MRx
        self.transport_log = np.array([[0,0,0]] * 24)   #For MRx

        self.starting_nodes = const.choose_starting_nodes()

        self.turn_number = 0 # Retains the turn number that has been completed
        self.turn_sub_counter = 0 #Retains whose turn it is during the specific turn
                                  #0 - MRx, [1->5] for detectives iteratively
        self.complete = False
        self.reward = 0
        self.MRx_moves = True
        self.detective_moves = [True,True,True,True,True] #If any detective is in a location where he does
                                                          #not have the token to move away from, it is set to False
        #self.log_file = None
        #For metadata and debugging, we will need more data. The following variables will store metadata

        self.detectives_transport_log = np.array([self.transport_log] * 5)
        self.detectives_location_log = np.array([self.locations] * 5)

    def initialize_game(self):

        self.log_file = open('./log/logs.txt','a')
        self.G = g_util.make_graph()
        for i in range(5):
            self.detectives[i] = d_util.initialize_detective( self.starting_nodes[i+1] )
        self.MRx = x_util.initialize_x(self.starting_nodes[0])
        self.reward = 0
        #self.log_start()

    def log_start(self):

        self.log_file.write('Staring New Game \n')
        self.log_file.write('\n')

        self.log_file.write('Starting Nodes: Mr_X Detective [0:4] ')
        self.log_file.write(str(self.starting_nodes))


    def take_action(self,next_node,mode):
        if self.complete == True:
            print('Game Over. Please initialize_game again.')
            return
        if self.turn_sub_counter == 0:
            self.play_MRx(next_node,mode)
        else:
            self.play_detective(next_node,mode)

        self.turn_sub_counter = self.turn_sub_counter + 1
        if self.turn_sub_counter == 6:
            self.turn_sub_counter = 0
            self.turn_number = self.turn_number + 1

        self.update()
        if self.complete == True:
            self.end_game()

    def play_MRx(self,next_node,mode):
        self.MRx = x_util.move_x(self.MRx,next_node,mode)
        self.locations[self.turn_number] = next_node
        self.transport_log[self.turn_number] = mode

    def play_detective(self,next_node,mode):
        self.detectives[self.turn_sub_counter - 1] = d_util.move_detective(self.detectives[self.turn_sub_counter - 1],next_node,mode)
        self.detectives_transport_log[self.turn_sub_counter - 1][self.turn_number] = mode
        self.detectives_location_log[self.turn_sub_counter - 1][self.turn_number] = next_node

    def update(self):
        #Can MRx Make any moves?
        pass #complete function in MRx
        #Can detectives make any moves?

        #If all Detectives or MRx cannot move, game is complete

    def end_game(self):
        pass
