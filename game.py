import numpy as np
import networkx as nx
import utilities.graph_utils as g_util #Graph utilities
import utilities.detective as d_util #detective's utilities
import utilities.mrx as x_util #MR X's utilities
import utilities.const as const
import time
import os

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
        self.reward = 0 #reward is as per detectives point of view
        self.MRx_moves = True
        self.detective_moves = [True,True,True,True,True] #If any detective is in a location where he does
                                                          #not have the token to move away from, it is set to False
        #For metadata and debugging, we will need more data. The following variables will store metadata

        self.detectives_transport_log = np.array([self.transport_log] * 5)
        self.detectives_location_log = np.array([self.locations] * 5)
        self.last_move_by_which_player = 0

        self.long_path = 10 # Hard coded to longest possible path calculated using longest_path() function after one run.
#Done

    def initialize_game(self):

        self.G = g_util.make_graph()
        self.starting_nodes = const.choose_starting_nodes()
        for i in range(5):
            self.detectives[i] = d_util.initialize_detective( self.starting_nodes[i+1] )
        self.MRx = x_util.initialize_x(self.starting_nodes[0])
        self.reward = 0
        self.complete = False

        self.locations = np.array([0] * 24)   #For MRx
        self.transport_log = np.array([[0,0,0]] * 24)   #For MRx

        self.turn_number = 0 # Retains the turn number that has been completed
        self.turn_sub_counter = 0 #Retains whose turn it is during the specific turn
                                  #0 - MRx, [1->5] for detectives iteratively


        self.MRx_moves = True
        self.detective_moves = [True,True,True,True,True] #If any detective is in a location where he does
                                                          #not have the token to move away from, it is set to False

        self.detectives_transport_log = np.array([self.transport_log] * 5)
        self.detectives_location_log = np.array([self.locations] * 5)
        self.last_move_by_which_player = 0

        #self.long_path = self.longest_path()

    def valid_moves(self):
        if self.turn_sub_counter == 0:
            return x_util.x_valid_list(self.MRx,self.G)
        else:
            return d_util.dec_valid_list(self.detectives,self.G,self.turn_sub_counter - 1)

    def end_turn_valid_moves(self):

        #print('Moves for ',self.last_move_by_which_player)
        if self.last_move_by_which_player == 0:
            return x_util.x_valid_list(self.MRx,self.G)
        else:
            return d_util.dec_valid_list(self.detectives,self.G,self.last_move_by_which_player - 1)


    def take_action(self,next_node,mode):
        if self.complete == True:
            print('Game Over. Please call initialize_game again.')
            return
        if self.turn_sub_counter == 0:
            self.play_MRx(next_node,mode)
        else:
            self.play_detective(next_node,mode)

        self.update()

        observation,_ = self.observe()

        self.last_move_by_which_player = self.turn_sub_counter
        self.turn_sub_counter = self.turn_sub_counter + 1
        if self.turn_sub_counter == 6: #1 turn over
            self.turn_sub_counter = 0
            self.turn_number = self.turn_number + 1

        self.skip_turn()

        return observation,self.reward,self.complete

    def skip_turn(self):
        if self.turn_sub_counter == 0:
            if self.MRx_moves == False:
                self.turn_sub_counter = self.turn_sub_counter + 1
                self.update()
                if self.complete == False:
                    self.skip_turn()
        else:
            if self.detective_moves[self.turn_sub_counter - 1] == False:
                self.turn_sub_counter = self.turn_sub_counter + 1
                if self.turn_sub_counter == 6:
                    self.turn_sub_counter = 0
                    self.turn_number = self.turn_number + 1
                self.update()
                if self.complete == False:
                    self.skip_turn()

        return

    def play_MRx(self,next_node,mode):
        self.MRx = x_util.move_x(self.MRx,next_node,mode)
        self.locations[self.turn_number] = next_node
        self.transport_log[self.turn_number] = mode

    def play_detective(self,next_node,mode):
        self.detectives[self.turn_sub_counter - 1] = d_util.move_detective(self.detectives[self.turn_sub_counter - 1],next_node,mode)
        self.detectives_transport_log[self.turn_sub_counter - 1][self.turn_number] = mode
        self.detectives_location_log[self.turn_sub_counter - 1][self.turn_number] = next_node
        #Sending tokens to MRx
        self.MRx = d_util.send_token(self.MRx,mode)

    def metadata_current_position(self):
        if self.turn_sub_counter == 0:
            return self.MRx[0]
        else:
            return self.detectives[self.turn_sub_counter - 1][0]

    def update(self):

        if (x_util.x_valid_list(self.MRx,self.G).size == 0):
            self.MRx_moves = False
        else:
            self.MRx_moves = True

        all_detectives_cant_move = True

        for i in range(5):
            if (d_util.dec_valid_list(self.detectives,self.G,i).size == 0):
                self.detective_moves[i] = False
            else:
                all_detectives_cant_move = False
                self.detective_moves[i] = True

        assumption = False #If MRx and detective are on same node. Game is Over

        for detective in self.detectives:
            if self.MRx[0] == detective[0]:
                assumption = True
                break
        if assumption == True:
            self.complete = True
            self.reward = +100

        if all_detectives_cant_move == True: #IF all detectives cant move game is Over
            self.complete = True
            self.reward = -100

        if self.turn_number == 22: #Redundant has tokens will be exhausted by then
            self.compelete = True
            self.reward = -100


        #IF turn_22 is reached game is Over

    def observe(self):
        if self.turn_sub_counter == 0:
            return self.observe_as_mrx(),self.turn_sub_counter
        else:
            return self.observe_as_detective(),self.turn_sub_counter

    def observe_as_mrx(self):
        #Observations are -
        # 1) MRX's location
        # 2) MRX's tokens
        # 3) Locations of detective
        # 4) Tokens of detective
        # 5) turn Number

        observation = []
        observation = observation + g_util.node_one_hot(self.MRx[0])
        observation = observation + self.MRx[1:].tolist()

        observation = observation + g_util.node_one_hot(self.detectives[0][0])
        observation = observation + g_util.node_one_hot(self.detectives[1][0])
        observation = observation + g_util.node_one_hot(self.detectives[2][0])
        observation = observation + g_util.node_one_hot(self.detectives[3][0])
        observation = observation + g_util.node_one_hot(self.detectives[4][0])

        observation = observation + self.detectives[0][1:].tolist()
        observation = observation + self.detectives[1][1:].tolist()
        observation = observation + self.detectives[2][1:].tolist()
        observation = observation + self.detectives[3][1:].tolist()
        observation = observation + self.detectives[4][1:].tolist()

        observation = observation + [self.turn_number]
        return np.array(observation)

    def observe_as_detective(self):
        observation = []

        last_reveal_turn = 0
        for surface_point in const.surface_points:
            if surface_point > self.turn_number:
                break
            if last_reveal_turn < surface_point:
                last_reveal_turn = surface_point
        if last_reveal_turn != 0:
            observation = observation + g_util.node_one_hot(last_reveal_turn)
        else:
            observation = observation + g_util.node_one_hot_zero()

        observation = observation + [const.turn_to_next_reveal[self.turn_number]]

        detective_turn = np.array([0,0,0,0,0])
        detective_turn[self.turn_sub_counter - 1] = 1

        for i in range(5):
            if self.turn_sub_counter - 1 == i:
                observation = observation + [1]
            else:
                observation = observation + [0]

        observation = observation + g_util.node_one_hot(self.detectives[0][0])
        observation = observation + g_util.node_one_hot(self.detectives[1][0])
        observation = observation + g_util.node_one_hot(self.detectives[2][0])
        observation = observation + g_util.node_one_hot(self.detectives[3][0])
        observation = observation + g_util.node_one_hot(self.detectives[4][0])

        observation = observation + self.detectives[0][1:].tolist()
        observation = observation + self.detectives[1][1:].tolist()
        observation = observation + self.detectives[2][1:].tolist()
        observation = observation + self.detectives[3][1:].tolist()
        observation = observation + self.detectives[4][1:].tolist()

        for i in range(self.turn_number - 5, self.turn_number):
            observation = observation + self.transport_log[i].tolist()

        return np.array(observation)

    ## Used to calculate the shortest path between detectives and X when game is done.
    def shortest_path(self,number):
        length = nx.dijkstra_path_length(self.G,self.detectives[number][0],self.MRx[0])
        return length

    ## Used to calculate the longest possible distance between all the nodes. Takes too long to calculate so hard coded after one run.
    # For 200 node graph longest path = 10
    def longest_path(self):
        long_path = 0
        for i in range(1,200):
            for j in range(i,200):
                path = nx.dijkstra_path_length(self.G,i,j)
                if (path > long_path):
                    long_path = path
        return long_path
