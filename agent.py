import numpy as np
import os
import sys
import tensorflow as tf
import utilities.graph_utils as g_util
import rl_backend.modelX as mdX
import rl_backend.modelDetective as mdD
import game

class RunAgent(object):
    def __init__(self):
        self.mdx = mdX.Model()
        self.mdd = mdD.Model()
        self.q_target = []

    def run_agent(self,number):
        for i in range(number):
            self.start_agent()

    def start_agent(self,epsilon_start = 1.0,epsilon_end = 0,epsilon_decay_steps = 5000):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = np.linspace(self.epsilon_start, self.epsilon_end, num = self.epsilon_decay_steps)
        self.explore = 0

        for i in range(self.epsilon_decay_steps):
            print('Running Episode: ',i,end = '\t')
            self.explore = i
            self.run_episode()

    def run_episode(self):
        self.SL = game.ScotlandYard()
        self.SL.initialize_game()
        done = False
        while done == False:
            present_observation,sub_turn = self.SL.observe()
            actions = self.SL.valid_moves()

            if sub_turn == 0:
                optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdx)
            else:
                optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdd)

                #Have optimum_action

            A = np.ones(actions.shape[0], dtype=float) * self.epsilon[self.explore] / actions.shape[0]
            A[optimum_action] += (1.0 - self.epsilon[self.explore])
            print('Explore = ',self.epsilon[self.explore])
            action_probs = A
            taken_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_node = actions[taken_action][1]
            mode = actions[taken_action][2:]
            state_used = present_observation.tolist() + g_util.node_one_hot(next_node) + mode.tolist()
            next_observation,reward,done = self.SL.take_action(next_node,mode)
            actions = self.SL.end_turn_valid_moves()

            if sub_turn == 0:
                _,Q_max = self.getOptimum_Action(next_observation,actions,self.mdx)
                q_target = [Q_max - reward]
                #print ('Q_traget',q_target)
                self.mdx.optimize([state_used], [q_target])
            else:
                _,Q_max = self.getOptimum_Action(next_observation,actions,self.mdd)
                q_target = [Q_max + reward]
                    #print ('Q_traget',q_target)
                self.mdd.optimize([state_used], [q_target])

        #if sub_turn == 0:
            #q_target = reward
            #self.mdd.optimize([state_used])


    def getOptimum_Action(self,present_state,actions,model):
        if actions.shape[0] == 0:
            return -1,0
        #print('Actions',actions)
        Q_values = np.zeros(actions.shape[0])
        for i in range(actions.shape[0]):

            next_node = g_util.node_one_hot(actions[i][1])

            observation = present_state.tolist() + next_node + actions[i][2:].tolist()
            Q_values[i] = model.predict([observation])
        return np.argmax(Q_values),np.amax(Q_values)
