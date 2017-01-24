import numpy as np
import os
import time
import tensorflow as tf
import utilities.graph_utils as g_util
import rl_backend.modelX as mdX
import rl_backend.modelDetective as mdD
import game

class RunAgent(object):
    def __init__(self):
        self.mdx = mdX.Model()
        self.mdd = [None] * 6
        for i in range (1,6):
            self.mdd[i] = mdD.Model()
        self.q_target = []
        self.d_last_obs = []
        self.x_last_obs = []

        self.log_path = './log'
        _,month,day,hour,minute,_,_,_,_ = time.localtime(time.time())
        self.log_path = self.log_path + '/agentlog/' + str(month) + '-' + str(day) + '/'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_path = self.log_path +  str(hour) + 'o' + str(minute) + '.txt'
        print(self.log_path)

        self.log_file = open(self.log_path,'w')
        self.log_file.write('episode_num,explore,result,detective_wins,x_wins\n')
        self.log_file.close()
        self.run_number = 0
        self.reward = 0
        self.detective_win = 0
        self.x_win = 0

    def start_agent(self,epsilon_decay_steps = 5000):
        self.epsilon_start = 1.0
        self.epsilon_end = 0.0
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon = np.linspace(self.epsilon_start, self.epsilon_end, num = self.epsilon_decay_steps)
        self.explore = 0


        for i in range(self.epsilon_decay_steps):
            print('\nDetectives: ',self.detective_win,'\t\tMR X: ',self.x_win)
            print('Running Episode: ',i,end = '\t')
            self.explore = i
            print('Explore = ',self.epsilon[self.explore],end = '\t')
            print('Agent Path :',self.log_path,end = '\t')
            self.run_episode()
            self.run_number = self.run_number + 1
            self.log_file = open(self.log_path,'a')
            self.run_log()
            self.log_file.close()
            # Saving model every x = 20 episodes
            if (self.run_number % 20 == 0):
                for i in range(1,6):
                    self.mdd[i].save(episode = self.run_number)
                self.mdx.save(episode = self.run_number)


    def run_log(self):
        self.log_file.write(str(self.run_number))
        self.log_file.write(',')
        self.log_file.write(str(self.epsilon[self.explore]))
        self.log_file.write(',')
        self.log_file.write(str(self.reward))
        self.log_file.write(',')
        self.log_file.write(str(self.detective_win))
        self.log_file.write(',')
        self.log_file.write(str(self.x_win))
        self.log_file.write('\n')

    def run_episode(self):
        self.SL = game.ScotlandYard()
        self.SL.initialize_game()
        done = False
        import time
        import os
        while done == False:
            present_observation,sub_turn = self.SL.observe()
            actions = self.SL.valid_moves()

            if sub_turn == 0:
                optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdx)
            else:
                optimum_action,_ = self.getOptimum_Action(present_observation,actions,self.mdd[sub_turn])

                #Have optimum_action

            A = np.ones(actions.shape[0], dtype=float) * self.epsilon[self.explore] / actions.shape[0]
            A[optimum_action] += (1.0 - self.epsilon[self.explore])

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
                _,Q_max = self.getOptimum_Action(next_observation,actions,self.mdd[sub_turn])
                q_target = [Q_max + reward]
                    #print ('Q_traget',q_target)
                self.mdd[sub_turn].optimize([state_used], [q_target])
            if done == False:
                if sub_turn == 0:
                    self.x_last_obs = state_used
                elif sub_turn == 5:
                    self.d_last_obs = state_used

        self.reward = reward
        print('Reward: ',reward)
        if(reward == 10):
            self.detective_win = self.detective_win + 1
        else:
            self.x_win = self.x_win + 1
        if sub_turn == 0:
            for i in range(1,6):
                self.mdd[i].optimize([self.d_last_obs],[[Q_max]])
        else:
            self.mdx.optimize([self.x_last_obs],[[Q_max]])


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
