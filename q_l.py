import numpy as np
import os
import utilities.graph_utils as g_util
import game
import logging
import random as rd
import rl_backend.modelX
import rl_backend.modelDetective

class q_learn(object):
    def __init__(self,mdx,mdd,explore = 0.0,directory='test/',file_name='test.txt',loglevel='INFO'):
        log_path = './log/q_learning/' + directory
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_path = log_path + file_name
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger('simple_logger')
        hdlr_1 = logging.FileHandler(log_path)
        hdlr_1.setFormatter(formatter)
        #hdlr_1.setFormatter('%(asctime)s %(message)s')
        self.logger.setLevel(loglevel.upper())
        self.logger.addHandler(hdlr_1)
        self.directory = directory
        self.file_name = file_name
        self.loglevel = loglevel
        self.explore = explore
        self.logger.info('Starting Game - Explore : %s',str(self.explore))
        self.SL = game.ScotlandYard()
        self.mdx = mdx
        self.mdd = mdd
        self.x_last_obs = []
        self.d_last_obs = [[0] * 1427] * 5

    #def start_agent(self):
    #    start = 0
    #    for i in range(len(epsilon)):
    #        self.run_episode()
    #            start = start+1



    def run_episode(self):
        self.logger.info('Running Episode')
        self.SL.initialize_game(directory = self.directory,file_name = self.file_name, loglevel = self.loglevel)
        done = False
        while done == False:
            present_observation,sub_turn = self.SL.observe()
            self.logger.debug('Calling observation')
            self.logger.debug('present_observation: \n%s',str(present_observation))
            self.logger.info('sub_turn: %s',str(sub_turn))
            self.logger.debug('Shape of present_observation: %s',str(np.array(present_observation).shape))
            actions = self.SL.valid_moves()
            self.logger.debug('Calling Valid moves')
            self.logger.debug('Shape of valid_moves: %s',str(np.array(actions).shape))
            self.logger.debug('valid_actions: \n%s',str(actions))


            if sub_turn == 0:
                #optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdx)
                random = rd.randint(0, actions.shape[0]-1)
                next_node = actions[random][1]
                mode = actions[random][2:]
                next_observation,reward,done = self.SL.take_action(next_node,mode)
                continue
                self.logger.debug('Optimum_action for MRx: %s',str(optimum_action))
            else:
                optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdd[sub_turn - 1])
                self.logger.debug('Optimum_action for Detective %s : %s',str(sub_turn-1),str(optimum_action))

                #Have optimum_action

            action_probs = np.ones(actions.shape[0], dtype=float) * self.explore / actions.shape[0]
            action_probs[optimum_action] += (1.0 - self.explore)
            self.logger.debug('Action_probs: %s',str(action_probs))
            taken_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            self.logger.debug('Taken Action: %s',str(taken_action))
            if taken_action == optimum_action:
                self.logger.info('NO EXPLORING')
            else:
                self.logger.info('EXPLORING')

            next_node = actions[taken_action][1]
            self.logger.debug('next_node: %s',str(next_node))
            mode = actions[taken_action][2:]
            self.logger.debug('mode: %s',str(mode))
            state_used = present_observation.tolist() + g_util.node_one_hot(next_node) + mode.tolist()
            next_observation,reward,done = self.SL.take_action(next_node,mode)
            self.logger.debug('next_observation: \n%s',str(next_observation))
            self.logger.debug('Shape of next_observation: %s',str(np.array(next_observation).shape))
            self.logger.info('Reward: %s',str(reward))
            self.logger.info('Complete: %s',str(done))
            actions = self.SL.end_turn_valid_moves()
            self.logger.debug('New set of actions: \n%s',actions)
            self.logger.debug('Shape of new Set of Actions: %s',str(np.array(actions).shape))

            if sub_turn == 0:
                self.logger.debug('Getting MaxQ for MRx and Optimizing for MRx')
                ##_,Q_max = self.getOptimum_Action(next_observation,actions,self.mdx)
                ##q_target = [Q_max - reward]
                ##self.mdx.optimize([state_used], [q_target])

            else:
                self.logger.debug('Getting MaxQ for Detective %s and Optimizing for him/her',str(sub_turn - 1))
                _,Q_max = self.getOptimum_Action(next_observation,actions,self.mdd[sub_turn - 1])
                q_target = [Q_max + reward]
                self.mdd[sub_turn -1].optimize([state_used], [q_target])

            if done == False:
                if sub_turn == 0:
                    self.logger.debug('Saving this Observation as last used by MRx')
                    ##self.x_last_obs = state_used
                else:
                    self.logger.debug('Saving this Observation as last used by Detective %s',str(sub_turn - 1))
                    self.d_last_obs[sub_turn - 1] = state_used
            self.logger.info('\n\n')
        self.logger.info('LAST TURN: %s',str(sub_turn))
        self.reward = reward
        self.logger.info('Reward = %s',str(self.reward))
        if sub_turn == 0:
            self.logger.debug('Last Turn by MRx, Optimizing for detective')
            for i in range(5):
                self.mdd[i].optimize([self.d_last_obs[i]],[[self.reward]])
        else:
            self.logger.debug('Last Turn By Detective %s',str(sub_turn))
            self.logger.debug('Optimizing MRx')
            #self.mdx.optimize([self.x_last_obs],[[self.reward]])
            for i in range(5):
                if i == sub_turn - 1:
                    continue
                self.logger.debug('Optimizing Detective %s',str(i))
                self.mdd[i].optimize([self.d_last_obs[i]],[[self.reward]])
        self.SL.close_log() #Closing the log file
        return self.reward


    def getOptimum_Action(self,present_state,actions,model):
        self.logger.debug('In getOptimum_Action')
        self.logger.debug('Actions,Actions.shape:%s \n%s',str(np.array(actions).shape),str(actions))
        if actions.shape[0] == 0:
            self.logger.debug('No Actions Possible')
            return -30,0
        #print('Actions',actions)
        #Q_values = np.zeros(actions.shape[0])
        observation = [[]]*actions.shape[0]
        #print(len(observation))
        for i in range(actions.shape[0]):
            next_node = g_util.node_one_hot(actions[i][1])
            observation[i] = present_state.tolist() + next_node + actions[i][2:].tolist()
        Q_values = model.predict(observation)
        self.logger.debug('Q_values: %s',str(Q_values))
        self.logger.debug('Index,Max: %s %s',str(np.argmax(Q_values)),str(np.amax(Q_values)))
        return np.argmax(Q_values),np.amax(Q_values)

    def close_log(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
