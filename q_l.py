import numpy as np
import os
import utilities.graph_utils as g_util
import game
import logging
import random as rd

class q_learn(object):
    def __init__(self,mdx,mdd,explore = 0.0):

        self.SL = game.ScotlandYard()
        self.explore = explore

        self.mdx = mdx
        self.mdd = mdd #MDD passed is an array here.

        self.x_obs = []
        self.x_y = []
        self.detective_obs = []
        self.detective_y = []
        self.reward = 0

    def run_episode(self):
        self.SL.initialize_game()
        done = False
        while done == False:
            present_observation,sub_turn = self.SL.observe()
            actions = self.SL.valid_moves()
            if sub_turn == 0:
                optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdx)
            else:
                optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdd)

            action_probs = np.ones(actions.shape[0], dtype=float) * self.explore / actions.shape[0]
            action_probs[optimum_action] += (1.0 - self.explore)
            taken_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_node = actions[taken_action][1]
            mode = actions[taken_action][2:]

            state_used = present_observation.tolist() + g_util.node_one_hot(next_node) + mode.tolist()

            next_observation,reward,done = self.SL.take_action(next_node,mode)
            actions = self.SL.end_turn_valid_moves()

            if actions.shape[0] == 0:
                actions = np.array([[next_node,next_node,0,0,0]])


            if sub_turn == 0:
                _,Q_max = self.getOptimum_Action(next_observation,actions,self.mdx)
                q_target = [Q_max] ###REWARD NOT ADDED
                self.x_obs.append(state_used)
                self.x_y.append(q_target)

            else:
                _,Q_max = self.getOptimum_Action(next_observation,actions,self.mdd)
                q_target = [Q_max] ####REWARD NOT ADDED
                self.detective_obs.append(state_used)
                self.detective_y.append(q_target)
        self.reward = reward
        self.learn()
        return self.reward,self.mdx,self.mdd

    def learn(self):

        rnge = np.array(self.x_obs).shape[0]
        for i in range(rnge):
            multiplier = rnge - i
            self.x_y[i][0] = self.x_y[i][0] - ((0.9)**multiplier) * self.reward
        #self.x_y = np.array(self.x_y)
        #self.x_obs = np.array(self.x_obs)
        self.mdx.optimize(self.x_obs,self.x_y)

        rnge = np.array(self.detective_obs).shape[0]
        for i in range(rnge):
            multiplier = rnge - i
            self.detective_y[i][0] = self.detective_y[i][0] + ((0.9)**multiplier) * self.reward
        #self.detective_y = np.array(self.detective_y)
        #self.detective_obs = np.array(self.detective_obs)
        self.mdd.optimize(self.detective_obs,self.detective_y)




    def getOptimum_Action(self,present_state,actions,model):
        Q_values = np.zeros(actions.shape[0])
        observation = [[]]*actions.shape[0]
        for i in range(actions.shape[0]):
            next_node = g_util.node_one_hot(actions[i][1])
            observation[i] = present_state.tolist() + next_node + actions[i][2:].tolist()
        Q_values = model.predict(observation)
        return np.argmax(Q_values),np.amax(Q_values)
