import numpy as np
import os
import random
import sys
import tensorflow as tf
import utilities.graph_utils as g_util
import rl_backend.modelX as mdX
import rl_backend.modelDetective as mdD
import game

def deep_q_learning(num_episodes = 100,
                    epsilon_start = 1.0,
                    epsilon_end = 0.1,
                    epsilon_decay_steps = 5000
                        ):

    # The epsilopn decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # Making game environment
    SL = game.ScotlandYard()

    mdX.initialize()
    mdD.initialize()
    q_target = []
    for i in range(num_episodes):

        SL.initialize_game()
        done = False
        while done == False:
            state, sub_turn = SL.observe()
            actions = SL.valid_moves()
            # Choosing model based on whose turn it is
            if(sub_turn == 0):

                Q_values = np.zeros(actions.shape[0])
                for j in range(actions.shape[0]):
                    print('print j: ', j)
                    next_node = g_util.node_one_hot(actions[j][1])
                    observation = state.tolist() + next_node + actions[j][2:].tolist()
                    observation[1][:] = observation[:]
                    #observation[j] = np.append(state, values = [next_node, actions[j][2:]]) ##Check axis
                    #print (observation[j])

                    print('Shape: ',np.array(observation).shape)
                    Q_values[j] = mdX.predict(observation)

            else:
                Q_values = np.zeros(actions.shape[0])
                for j in range(actions.shape[0]):
                    next_node = g_util.node_one_hot(actions[j][1])
                    observation = state.tolist() + next_node + actions[j][2:].tolist()
                    #observation = np.append(state, values = [next_node, actions[j][2:]], axis = 0)
                    Q_values[j] = mdD.predict(observation)

            optimum_action = np.argmax(Q_values)

            used_state = observation[optimum_action]
            epsilon_decay_steps = epsilon_decay_steps - 1
            A = np.ones(nA, dtype=float) * epsilon[epsilon_decay_steps] / actions.shape[0]
            A[optimum_action] += (1.0 - epsilon[epsilon_decay_steps])

            action_probs = A
            taken_action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_node = actions[taken_action][1]
            mode = actions[taken_action][2:]

            next_state, reward, done = SL.take_action(next_node, mode)
            next_node = g_util.node_one_hot(actions[take_action][1])

            used_state = state.tolist() + next_node + actions[taken_action][2:].tolist()
            actions = SL.end_turn_valid_moves()



            if (sub_turn == 0):
                Q_values = np.zeros(actions.shape[0])
                for j in range(actions.shape[0]):
                    next_state = g_util.node_onehot(actions[j][1])
                    observation = state.tolist() + next_node + actions[j][2:].tolist()
                    Q_values[j] = mdX.predict(observation)

                optimum_action = np.argmax(Q_values)
                q_target = Q_values[optimum_action] + reward
                mdX.optimize(used_state, q_target)

            else:

                Q_values = np.zeros(actions.shape[0])
                for j in range(actions.shape[0]):
                    next_state = g_util.node_onehot(actions[j][1])
                    observation = state.tolist() + next_node + actions[j][2:].tolist()
                    Q_values[j] = mdD.predict(observation)

                optimum_action = np.argmax(Q_values)
                q_target = Q_values[optimum_action] + reward
                mdD.optimize(used_state, q_target)
