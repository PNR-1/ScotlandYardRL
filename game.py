import numpy as np
import networkx as nx
import utilities.graph_utils as g_util #Graph utilities
import utilities.detective as d_util #detective's utilities
import utilities.mrx as x_util #MR X's utilities
import utilities.const as const
import os
import logging

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

    def initialize_game(self,directory='test/',file_name='test.txt',loglevel='INFO'):

        self.loglevel = loglevel
        self.G = g_util.make_graph()
        self.starting_nodes = const.choose_starting_nodes()
        for i in range(5):
            self.detectives[i] = d_util.initialize_detective( self.starting_nodes[i+1] )
        self.MRx = x_util.initialize_x(self.starting_nodes[0])
        self.reward = 0
        self.log_start(directory,file_name)

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

        self.logger.debug('Initialize_Game Called')


    def valid_moves(self):
        self.logger.debug('Sending Valid Moves for %s',str(self.turn_sub_counter))
        if self.turn_sub_counter == 0:
            return x_util.x_valid_list(self.MRx,self.G)
        else:
            return d_util.dec_valid_list(self.detectives,self.G,self.turn_sub_counter - 1)

    def end_turn_valid_moves(self):

        #print('Moves for ',self.last_move_by_which_player)
        self.logger.debug('Sending Valid Moves for %s',str(self.last_move_by_which_player))
        if self.last_move_by_which_player == 0:
            return x_util.x_valid_list(self.MRx,self.G)
        else:
            return d_util.dec_valid_list(self.detectives,self.G,self.last_move_by_which_player - 1)


    def take_action(self,next_node,mode):
        self.logger.debug('take_action called')
        self.logger.debug('Turn Number = %s, Sub Counter = %s',str(self.turn_number),str(self.turn_sub_counter))
        self.logger.debug('Values Given - next_node = %s, mode = %s',str(next_node),str(mode))
        if self.complete == True:
            self.logger.info('Game Complete has been reached at start of take_action')
            print('Game Over. Please call initialize_game again.')
            return
        if self.turn_sub_counter == 0:
            self.logger.debug('turn_sub_counter = 0, Playing Mrx with values')
            self.play_MRx(next_node,mode)
        else:
            self.logger.debug('turn_sub_counter = %s, Playing respective detective',str(self.turn_sub_counter))
            self.play_detective(next_node,mode)
        self.logger.debug('Calling Update')
        self.update()
        self.logger.debug('Update Over')
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

        return observation,self.reward,self.complete

    def skip_turn(self):
        self.logger.debug('Skip Turn')
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
        self.logger.debug('Reached play_MRx, next_node = %s, mode = %s',str(next_node),str(mode))
        self.logger.debug('Before Move-> MRx: %s',str(self.MRx))
        self.MRx = x_util.move_x(self.MRx,next_node,mode)
        self.logger.debug('After Move-> MRx: %s',str(self.MRx))
        self.locations[self.turn_number] = next_node
        self.transport_log[self.turn_number] = mode

    def play_detective(self,next_node,mode):
        self.logger.debug('Reached play_detective, next_node = %s, mode = %s, turn_sub_counter = %s',str(next_node),str(mode),str(self.turn_sub_counter))
        self.logger.debug('Before Move-> Detective[%s]: %s',str(self.turn_sub_counter - 1), str(self.detectives[self.turn_sub_counter - 1]))
        self.detectives[self.turn_sub_counter - 1] = d_util.move_detective(self.detectives[self.turn_sub_counter - 1],next_node,mode)
        self.logger.debug('After Move-> Detective[%s]: %s',str(self.turn_sub_counter - 1), str(self.detectives[self.turn_sub_counter - 1]))
        self.detectives_transport_log[self.turn_sub_counter - 1][self.turn_number] = mode
        self.detectives_location_log[self.turn_sub_counter - 1][self.turn_number] = next_node
        #Sending tokens to MRx
        self.logger.debug('Sending Tokens to MRx')
        self.logger.debug('Before Sending-> MRx: %s',str(self.MRx))
        self.MRx = d_util.send_token(self.MRx,mode)
        self.logger.debug('After Sending-> MRx: %s',str(self.MRx))

    def update(self):
        self.logger.debug('Valid Moves for MRx: %s',x_util.x_valid_list(self.MRx,self.G))
        if (x_util.x_valid_list(self.MRx,self.G).size == 0):
            self.logger.debug('MRx has no valid Moves')
            self.MRx_moves = False
        else:
            self.logger.debug('MRx has valid moves')
            self.MRx_moves = True

        all_detectives_cant_move = True

        for i in range(5):
            self.logger.debug('Detective %s\'s valid move: %s',str(i),d_util.dec_valid_list(self.detectives,self.G,i))
            if (d_util.dec_valid_list(self.detectives,self.G,i).size == 0):
                self.logger.debug('Detective cant move')
                self.detective_moves[i] = False
            else:
                self.logger.debug('Detective can move')
                all_detectives_cant_move = False
                self.detective_moves[i] = True

        assumption = False #If MRx and detective are on same node. Game is Over

        for detective in self.detectives:
            if self.MRx[0] == detective[0]:
                self.logger.debug('Detective on MRx\' position')
                assumption = True
                break
        if assumption == True:
            self.logger.info('Detetives won')
            self.complete = True
            self.reward = +100

        if all_detectives_cant_move == True: #IF all detectives cant move game is Over
            self.logger.info('ALL Detectives cant move')
            self.complete = True
            self.reward = -100

        if self.turn_number == 23: #Redundant has tokens will be exhausted by then
            self.logger.info('Tokens over')
            self.compelete = True
            self.reward = -100


        #IF turn_22 is reached game is Over

    def observe(self):
        self.logger.debug('Observation Called')
        self.logger.debug('Turn Number = %s, Sub_Counter = %s',str(self.turn_number),str(self.turn_sub_counter))
        if self.turn_sub_counter == 0:
            self.logger.debug('Calling MRx\'s observation')
            return self.observe_as_mrx(),self.turn_sub_counter
        else:
            self.logger.debug('Calling Detective Observation')
            return self.observe_as_detective(),self.turn_sub_counter

    def observe_as_mrx(self):
        #Observations are -
        # 1) MRX's location
        # 2) MRX's tokens
        # 3) Locations of detective
        # 4) Tokens of detective
        # 5) turn Number

        observation = []
        self.logger.debug('Creating MRx Observation')
        self.logger.debug('MRX-> %s',str(self.MRx))
        self.logger.debug('One Hot Encoding Value: %s',str(self.MRx[0]))
        observation = observation + g_util.node_one_hot(self.MRx[0])
        self.logger.debug('Mode Values: %s',str(self.MRx[1:].tolist()))
        observation = observation + self.MRx[1:].tolist()

        self.logger.debug('Adding Detective Positions')
        self.logger.debug('Detectives -> %s',str(self.detectives))
        self.logger.debug('Adding One Hot - %s %s %s %s %s',str(self.detectives[0][0]),str(self.detectives[1][0]),str(self.detectives[2][0]),str(self.detectives[3][0]),str(self.detectives[4][0]))
        observation = observation + g_util.node_one_hot(self.detectives[0][0])
        observation = observation + g_util.node_one_hot(self.detectives[1][0])
        observation = observation + g_util.node_one_hot(self.detectives[2][0])
        observation = observation + g_util.node_one_hot(self.detectives[3][0])
        observation = observation + g_util.node_one_hot(self.detectives[4][0])
        self.logger.debug('Adding modes - %s %s %s %s %s',str(self.detectives[0][1:].tolist()),str(self.detectives[1][1:].tolist()),str(self.detectives[2][1:].tolist()),str(self.detectives[3][1:].tolist()),str(self.detectives[4][1:].tolist()))
        observation = observation + self.detectives[0][1:].tolist()
        observation = observation + self.detectives[1][1:].tolist()
        observation = observation + self.detectives[2][1:].tolist()
        observation = observation + self.detectives[3][1:].tolist()
        observation = observation + self.detectives[4][1:].tolist()
        self.logger.debug('Adding turn number %s',str(self.turn_number))
        observation = observation + [self.turn_number]
        observation = np.array(observation)
        self.logger.debug('Shape of Observation = %s',str(observation.shape))
        return observation

    def observe_as_detective(self):
        observation = []
        self.logger.debug('Creating Detective Observation')
        last_reveal_turn = 0
        for surface_point in const.surface_points:
            if surface_point > self.turn_number:
                break
            if last_reveal_turn < surface_point:
                last_reveal_turn = surface_point
        self.logger.debug('Last Reveal Turn = %s',str(last_reveal_turn))
        if last_reveal_turn != 0:
            self.logger.debug('MRX Location Log - > %s',self.locations)
            self.logger.debug('Adding Node %s',str(self.locations[last_reveal_turn]))
            observation = observation + g_util.node_one_hot(self.locations[last_reveal_turn])
        else:
            self.logger.debug('Adding 0 node for MRx Location')
            observation = observation + g_util.node_one_hot_zero()

        observation = observation + [const.turn_to_next_reveal[self.turn_number]]
        self.logger.debug('Turn to next Reveal = %s',observation[-1])
        self.logger.debug('Adding One Hot - %s %s %s %s %s',str(self.detectives[0][0]),str(self.detectives[1][0]),str(self.detectives[2][0]),str(self.detectives[3][0]),str(self.detectives[4][0]))
        observation = observation + g_util.node_one_hot(self.detectives[0][0])
        observation = observation + g_util.node_one_hot(self.detectives[1][0])
        observation = observation + g_util.node_one_hot(self.detectives[2][0])
        observation = observation + g_util.node_one_hot(self.detectives[3][0])
        observation = observation + g_util.node_one_hot(self.detectives[4][0])
        self.logger.debug('Adding modes - %s %s %s %s %s',str(self.detectives[0][1:].tolist()),str(self.detectives[1][1:].tolist()),str(self.detectives[2][1:].tolist()),str(self.detectives[3][1:].tolist()),str(self.detectives[4][1:].tolist()))
        observation = observation + self.detectives[0][1:].tolist()
        observation = observation + self.detectives[1][1:].tolist()
        observation = observation + self.detectives[2][1:].tolist()
        observation = observation + self.detectives[3][1:].tolist()
        observation = observation + self.detectives[4][1:].tolist()

        self.logger.debug('MRx\'s Last transport: %s',str(self.transport_log))
        for i in range(self.turn_number - 4, self.turn_number+1):
            self.logger.debug('Adding : %s',str(self.transport_log[i].tolist()))
            observation = observation + self.transport_log[i].tolist()

        observation = np.array(observation)
        self.logger.debug('Shape of Observations %s',observation.shape)
        return observation

    def log_turn(self,next_node,mode):
        self.logger.info('Turn Number: %s',str(self.turn_number))
        self.logger.info('Turn Sub Counter: %s',str(self.turn_sub_counter))
        self.logger.info('Next Node: %s',str(next_node))
        self.logger.info('Mode: %s',str(mode))
        self.logger.info('MRx: %s',str(self.MRx))
        self.logger.info('Detectives: \n%s',str(self.detectives))
        self.logger.info('#######\n\n\n\n')

    def log_start(self,directory,file_name):
        log_path = './log/gamelog/' + directory
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = log_path +  file_name
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger('simple_logger')
        hdlr_1 = logging.FileHandler(log_path)
        hdlr_1.setFormatter(formatter)
        #hdlr_1.setFormatter('%(asctime)s %(message)s')
        self.logger.setLevel(self.loglevel.upper())
        self.logger.addHandler(hdlr_1)
        #self.logger.basicConfig(filename = log_path,level = self.loglevel.upper(),format= '%(asctime)s %(message)s' )

        self.logger.info('Starting New Game')
        self.logger.info('Starting Nodes: Mr_X, Detective [0:4] %s',str(self.starting_nodes))
        self.logger.info('MRx: %s',str(self.MRx))
        self.logger.info('Detectives:\n%s',str(self.detectives))
        self.logger.info('END OF INIT\n\n\n\n\n')

    def complete_log(self):
        self.logger.info('\n\n\n\n####### ###### END GAME')
        self.logger.info('Turn Number: %s',str(self.turn_number))
        self.logger.info('Turn Sub Counter: %s',str(self.turn_sub_counter))

        self.logger.info('MRx: %s',str(self.MRx))
        self.logger.info('MRx Locations:\n%s',str(self.locations))

        self.logger.info('MRx transport_log:\n%s',str(self.transport_log))

        self.logger.info('Does MRx have a valid move? %s',str(self.MRx_moves))

        self.logger.info('Detectives:\n%s',str(self.detectives))

        self.logger.info('Detectives Location Log:\n%s',str(self.detectives_location_log))

        #self.logger.info('Detective Transport Log \n%s',str(g_util.print_list(self.detectives_transport_log)))

        self.logger.info('Does a detective have a valid move? %s',str(self.detective_moves))

        self.logger.info('Complete: %s',str(self.complete))

        self.logger.info('Reward: %s',str(self.reward))

        self.logger.info('#######\n\n\n')
    def close_log(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
