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
        self.log_file = None
        #For metadata and debugging, we will need more data. The following variables will store metadata

        self.detectives_transport_log = np.array([self.transport_log] * 5)
        self.detectives_location_log = np.array([self.locations] * 5)
        self.last_move_by_which_player = 0
#Done

    def initialize_game(self):

        self.G = g_util.make_graph()
        self.starting_nodes = const.choose_starting_nodes()
        for i in range(5):
            self.detectives[i] = d_util.initialize_detective( self.starting_nodes[i+1] )
        self.MRx = x_util.initialize_x(self.starting_nodes[0])
        self.reward = 0
        self.log_start()
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
        self.log_turn(next_node,mode)

        observation,_ = self.observe()

        self.last_move_by_which_player = self.turn_sub_counter
        self.turn_sub_counter = self.turn_sub_counter + 1
        if self.turn_sub_counter == 6: #1 turn over
            self.turn_sub_counter = 0
            self.turn_number = self.turn_number + 1

        self.skip_turn()

        if self.complete == True:
            self.complete_log()
            self.log_file.close()

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
            self.reward = +10

        if all_detectives_cant_move == True: #IF all detectives cant move game is Over
            self.complete = True
            self.reward = -10

        if self.turn_number == 22: #Redundant has tokens will be exhausted by then
            self.compelete = True
            self.reward = -10


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

    def log_turn(self,next_node,mode):
        self.log_file.write('Turn Number: ')
        self.log_file.write(str(self.turn_number))
        self.log_file.write('\t\t Turn Sub Counter: ')
        self.log_file.write(str(self.turn_sub_counter))
        self.log_file.write('\n')
        self.log_file.write('Next Node: ')
        self.log_file.write(str(next_node))
        self.log_file.write('\n')
        self.log_file.write('Mode: ')
        self.log_file.write(str(mode))
        self.log_file.write('\n')
        self.log_file.write('MRx: ')
        self.log_file.write(str(self.MRx))
        self.log_file.write('\n')
        self.log_file.write('Detectives: ')
        self.log_file.write(str(self.detectives))
        self.log_file.write('\n')
        self.log_file.write('#######')
        self.log_file.write('\n')

    def log_start(self):
        log_path = './log'
        _,month,day,hour,minute,second,_,_,_ = time.localtime(time.time())
        log_path = log_path + '/gamelog/' + str(month) + '-' + str(day) + '/' + str(hour) + '/'
        print(log_path)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = log_path +  str(minute) + 'o' + str(second) + '.txt'
        self.log_file = open(log_path,'w')

        self.log_file.write('Staring New Game \n')
        self.log_file.write('\n')
        self.log_file.write('Starting Nodes: Mr_X, Detective [0:4] ')
        self.log_file.write(str(self.starting_nodes))
        self.log_file.write('\n')
        self.log_file.write('MRx: ')
        self.log_file.write(str(self.MRx))
        self.log_file.write('\n')
        self.log_file.write('MRx Locations: ')
        self.log_file.write(str(self.locations))
        self.log_file.write('\n')
        self.log_file.write('Detectives: ')
        self.log_file.write(str(self.detectives))
        self.log_file.write('\n')
        self.log_file.write('\t\t\t\t#######')
        self.log_file.write('\n')
        self.log_file.write('\n')

    def complete_log(self):
        self.log_file.write('\n\n ####### ###### END GAME\n')
        self.log_file.write('Turn Number: ')
        self.log_file.write(str(self.turn_number))
        self.log_file.write('\t\t Turn Sub Counter: ')
        self.log_file.write(str(self.turn_sub_counter))
        self.log_file.write('\n')

        self.log_file.write('MRx: ')
        self.log_file.write(str(self.MRx))
        self.log_file.write('\n')

        self.log_file.write('MRx Locations:\n')
        self.log_file.write(str(self.locations))
        self.log_file.write('\n')

        self.log_file.write('MRx transport_log:\n')
        self.log_file.write(str(self.transport_log))
        self.log_file.write('\n')

        self.log_file.write('Does MRx have a valid move? ')
        self.log_file.write(str(self.MRx_moves))
        self.log_file.write('\n')

        self.log_file.write('Detectives:\n')
        self.log_file.write(str(self.detectives))
        self.log_file.write('\n')

        self.log_file.write('Detectives Location Log:\n')
        self.log_file.write(str(self.detectives_location_log))
        self.log_file.write('\n')

        g_util.print_list(self.log_file,self.detectives_transport_log)

        self.log_file.write('Does a detective have a valid move? ')
        self.log_file.write(str(self.detective_moves))
        self.log_file.write('\n')

        self.log_file.write('Complete: ')
        self.log_file.write(str(self.complete))
        self.log_file.write('\n')

        self.log_file.write('Reward: ')
        self.log_file.write(str(self.reward))
        self.log_file.write('\n')

        self.log_file.write('#######')
